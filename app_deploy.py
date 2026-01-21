import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, storage
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import json
import os
import re
import time
import google.generativeai as genai
from PIL import Image
from streamlit_cropper import st_cropper

# ==========================================
# 0. Global Constants
# ==========================================
BUCKET_NAME = "math-problem-collector.firebasestorage.app"
TEMP_DIR = "temp_images"

# Ensure temp directory exists for intermediate processing if needed
os.makedirs(TEMP_DIR, exist_ok=True)

# ==========================================
# 1. Configuration & Caching (The Backbone)
# ==========================================
st.set_page_config(layout="wide", page_title="Cloud Math Labeler")

@st.cache_resource
def init_firebase():
    """
    Initializes Firebase only once per session. 
    Uses st.cache_resource to prevent re-initialization on every rerun.
    """
    try:
        if not firebase_admin._apps:
            if "firebase" in st.secrets:
                cred = credentials.Certificate(dict(st.secrets["firebase"]))
            elif os.path.exists("serviceAccountKey.json"):
                cred = credentials.Certificate("serviceAccountKey.json")
            else:
                return None, None
            
            app = firebase_admin.initialize_app(cred, {'storageBucket': BUCKET_NAME})
            return firestore.client(), storage.bucket()
        else:
            return firestore.client(), storage.bucket()
    except Exception as e:
        st.error(f"🔥 Firebase Init Error: {e}")
        return None, None

@st.cache_resource
def get_drive_service():
    """
    Authenticates Google Drive API once and caches the service object.
    Patches missing token_uri to prevent 'No access token' errors.
    """
    SCOPES = ['https://www.googleapis.com/auth/drive']
    creds = None
    
    try:
        if "firebase" in st.secrets:
            # 1. Load the secrets as a mutable dictionary
            key_dict = dict(st.secrets["firebase"])
            
            # 2. CRITICAL FIX: Force the token_uri if missing
            # This tells google-auth where to exchange the JWT for an Access Token
            if "token_uri" not in key_dict:
                key_dict["token_uri"] = "https://oauth2.googleapis.com/token"
            
            creds = service_account.Credentials.from_service_account_info(
                key_dict, scopes=SCOPES
            )
        elif os.path.exists("serviceAccountKey.json"):
            creds = service_account.Credentials.from_service_account_file(
                "serviceAccountKey.json", scopes=SCOPES
            )
        
        if creds:
            return build('drive', 'v3', credentials=creds)
        return None
    except Exception as e:
        st.error(f"🚗 Drive Auth Error: {e}")
        return None

# Initialize resources
db, bucket = init_firebase()
drive_service = get_drive_service()

if not db or not drive_service:
    st.error("❌ Critical System Failure: Auth credentials missing or invalid.")
    st.stop()

# ==========================================
# 2. Logic & Data Processing
# ==========================================

OPTIONS = {
    "subject": ["수학II", "수학I", "미적분", "확률과통계", "기하", "공통수학"],
    "grade": ["고2", "고1", "고3", "N수", "중등"],
    "unit_major": [
        "함수의 극한과 연속", "미분법", "적분법", 
        "지수함수와 로그함수", "삼각함수", "수열",
        "순열과 조합", "확률", "통계",
        "이차곡선", "평면벡터", "공간도형과 공간좌표",
        "다항식", "방정식과 부등식", "행렬", "기타"
    ],
    "difficulty": ["상", "최상(Killer)", "중", "하", "최하"],
    "question_type": ["추론형", "계산형", "이해형", "문제해결형", "합답형"],
    "source_org": ["평가원", "교육청", "사관학교/경찰대", "EBS", "내신", "기타"],
    "concepts": ["샌드위치 정리", "절댓값 함수", "미분계수의 정의", "평균값 정리", "롤의 정리", "사이값 정리", "극대/극소", "변곡점", "정적분 정의", "부분적분", "치환적분", "도함수 활용", "기타"] 
}

def list_drive_images(folder_id):
    try:
        query = f"'{folder_id}' in parents and (mimeType contains 'image/') and trashed = false"
        results = drive_service.files().list(
            q=query, 
            fields="files(id, name)", 
            pageSize=100
        ).execute()
        return results.get('files', [])
    except Exception as e:
        st.error(f"Error listing files: {e}")
        return []

def download_image_from_drive(file_id):
    try:
        request = drive_service.files().get_media(fileId=file_id)
        file_obj = io.BytesIO()
        downloader = MediaIoBaseDownload(file_obj, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        file_obj.seek(0)
        return Image.open(file_obj)
    except Exception as e:
        st.error(f"Download failed: {e}")
        return None

def move_file_to_done(file_id, current_folder_id, done_folder_id):
    try:
        drive_service.files().update(
            fileId=file_id,
            addParents=done_folder_id,
            removeParents=current_folder_id,
            fields='id, parents'
        ).execute()
        return True
    except Exception as e:
        st.error(f"Move failed: {e}")
        return False

def upload_image_to_storage(image, filename):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    path = f"cropped_problems/{filename}"
    blob = bucket.blob(path)
    blob.upload_from_string(img_byte_arr, content_type='image/jpeg')
    blob.make_public()
    return blob.public_url

def extract_gemini(image):
    if "GEMINI_API_KEY" not in st.secrets:
        return {"error": "API Key Missing in Secrets"}

    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-2.0-flash") # Updated to 2.0 for better speed/cost
        
        prompt = """
        Analyze this math problem image.
        1. Identify the mathematical equations and problem text.
        2. Describe any diagrams or graphs strictly.
        3. Return ONLY a valid JSON object with keys: "problem_text" (use LaTeX for math) and "diagram_desc".
        """
        
        # Configure generation config to force valid JSON if model supports it, 
        # otherwise rely on regex.
        response = model.generate_content([prompt, image])
        text = response.text
        
        # Robust Regex Parsing for JSON
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            clean_json = json_match.group(0)
            return json.loads(clean_json)
        else:
            # Fallback if no JSON found
            return {"problem_text": text, "diagram_desc": "Auto-extraction failed format check."}
            
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 3. UI Layout
# ==========================================
st.title("✂️ Cloud Math Cropper & Labeler (Pro)")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    default_folder = st.secrets.get("DEFAULT_FOLDER_ID", "")
    done_folder_default = st.secrets.get("DONE_FOLDER_ID", "")
    
    folder_id = st.text_input("Source Folder ID", value=default_folder)
    done_folder_id = st.text_input("Done Folder ID", value=done_folder_default)
    
    if st.button("📂 Load Drive Files", type="primary"):
        if folder_id:
            with st.spinner("Scanning Drive..."):
                files = list_drive_images(folder_id)
                st.session_state['drive_files'] = files
                st.session_state['idx'] = 0
                # Clear previous context
                st.session_state.pop('cropped_img', None)
                st.session_state.pop('extracted', None)
                st.success(f"Found {len(files)} images.")
        else:
            st.warning("Please enter a Source Folder ID.")

    st.markdown("---")
    
    # Navigation
    c_prev, c_next = st.columns(2)
    with c_prev:
        if st.button("◀ Prev"):
            if st.session_state.get('idx', 0) > 0:
                st.session_state['idx'] -= 1
                st.session_state.pop('cropped_img', None)
                st.session_state.pop('extracted', None)
                st.rerun()
                
    with c_next:
        if st.button("Next ▶"):
            files = st.session_state.get('drive_files', [])
            if files and st.session_state['idx'] < len(files) - 1:
                st.session_state['idx'] += 1
                st.session_state.pop('cropped_img', None)
                st.session_state.pop('extracted', None)
                st.rerun()

# ==========================================
# 4. Main Workspace
# ==========================================
if 'drive_files' in st.session_state and st.session_state['drive_files']:
    files = st.session_state['drive_files']
    idx = st.session_state['idx']
    
    # Safe Index Check
    if idx >= len(files):
        st.warning("Index out of range (files might have been moved). Resetting...")
        st.session_state['idx'] = 0
        st.rerun()
        
    current_file = files[idx]
    
    st.subheader(f"🖼️ [{idx+1}/{len(files)}] {current_file['name']}")
    
    # Load Image (Cached in session state to avoid re-download on interaction)
    if 'current_file_id' not in st.session_state or st.session_state['current_file_id'] != current_file['id']:
        with st.spinner("Downloading image..."):
            img = download_image_from_drive(current_file['id'])
            if img:
                st.session_state['original_img'] = img
                st.session_state['current_file_id'] = current_file['id']
                # Reset downstream states
                st.session_state.pop('cropped_img', None)
                st.session_state.pop('extracted', None)
            else:
                st.error("Failed to load image. Try next file.")
    
    if 'original_img' in st.session_state:
        # Cropper Section
        st.info("💡 Drag to crop the specific problem area.")
        
        # Realtime update is false to save resources, only updates on release/button usually better,
        # but kept true per your original preference.
        cropped_img = st_cropper(
            st.session_state['original_img'],
            realtime_update=True,
            box_color='#FF0000',
            aspect_ratio=None
        )
        
        col_view, col_action = st.columns([1, 1])
        
        with col_view:
            st.markdown("**Preview:**")
            st.image(cropped_img, use_container_width=True)
            
        with col_action:
            st.markdown("**AI Extraction:**")
            if st.button("✨ Analyze Selection", type="primary"):
                with st.spinner("Gemini is analyzing..."):
                    st.session_state['cropped_img'] = cropped_img
                    extracted_data = extract_gemini(cropped_img)
                    
                    if "error" in extracted_data:
                        st.error(extracted_data['error'])
                    else:
                        st.session_state['extracted'] = extracted_data
                        st.success("Analysis Complete")

    st.divider()

    # Data Entry Form
    if 'extracted' in st.session_state:
        item = st.session_state['extracted']
        
        with st.form("labeling_form"):
            st.subheader("📝 Metadata & Validation")
            
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            subject = r1c1.selectbox("Subject", OPTIONS['subject'])
            grade = r1c2.selectbox("Grade", OPTIONS['grade'])
            source = r1c3.selectbox("Source", OPTIONS['source_org'])
            unit = r1c4.selectbox("Unit", OPTIONS['unit_major'])
            
            r2c1, r2c2, r2c3 = st.columns(3)
            diff = r2c1.selectbox("Difficulty", OPTIONS['difficulty'])
            q_type = r2c2.selectbox("Type", OPTIONS['question_type'])
            concept = r2c3.selectbox("Key Concept", OPTIONS['concepts'])
            
            st.markdown("---")
            prob_text = st.text_area("Problem (LaTeX)", value=item.get('problem_text', ""), height=200)
            diag_desc = st.text_area("Diagram Description", value=item.get('diagram_desc', ""), height=100)
            
            submit_btn = st.form_submit_button("🔥 Save to Database & Move File")
            
            if submit_btn:
                if 'cropped_img' not in st.session_state:
                    st.error("Cropped image lost. Please crop again.")
                else:
                    try:
                        # 1. Upload Image
                        with st.spinner("Uploading to Storage..."):
                            timestamp = int(time.time())
                            clean_name = current_file['name'].rsplit('.', 1)[0]
                            # Sanitize filename
                            clean_name = re.sub(r'[^a-zA-Z0-9_-]', '', clean_name)
                            img_filename = f"{clean_name}_{timestamp}.jpg"
                            
                            img_url = upload_image_to_storage(st.session_state['cropped_img'], img_filename)
                        
                        # 2. Save Metadata
                        with st.spinner("Writing to Firestore..."):
                            doc_data = {
                                "original_filename": current_file['name'],
                                "drive_file_id": current_file['id'],
                                "image_url": img_url,
                                "storage_path": f"cropped_problems/{img_filename}",
                                "meta": {
                                    "subject": subject, "grade": grade, "source": source,
                                    "unit": unit, "difficulty": diff, "question_type": q_type,
                                    "concept": concept
                                },
                                "content": {"problem": prob_text, "diagram": diag_desc},
                                "created_at": firestore.SERVER_TIMESTAMP,
                                "labeler_version": "v2.0-optimized"
                            }
                            db.collection("math_dataset").add(doc_data)
                            
                        # 3. Move File
                        if done_folder_id:
                            with st.spinner("Archiving file..."):
                                success = move_file_to_done(current_file['id'], folder_id, done_folder_id)
                                if success:
                                    st.toast("✅ Saved & Archived!")
                                    # Update local list to reflect move
                                    st.session_state['drive_files'].pop(idx)
                                    # Cleanup state
                                    st.session_state.pop('cropped_img', None)
                                    st.session_state.pop('extracted', None)
                                    # No need to increment idx because the list shifted left
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("Saved data, but failed to move file in Drive.")
                        else:
                            st.warning("Data saved, but file not moved (No Done Folder ID provided).")
                            
                    except Exception as e:
                        st.error(f"Save failed: {e}")

else:
    st.info("👈 Connect to Google Drive from the sidebar to start.")

