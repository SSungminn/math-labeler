import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, storage
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import json
import os
import google.generativeai as genai
from PIL import Image
import time
from streamlit_drawable_canvas import st_canvas
import numpy as np # 리사이징 계산용

# ==========================================
# 0. 사용자 설정
# ==========================================
BUCKET_NAME = "math-problem-collector.firebasestorage.app"

# ==========================================
# 1. Configuration & Auth
# ==========================================
st.set_page_config(layout="wide", page_title="Math Labeling Studio")

def get_firebase_credentials():
    if "firebase" in st.secrets:
        return credentials.Certificate(dict(st.secrets["firebase"]))
    elif "serviceAccountKey.json" in [f.name for f in os.scandir('.')]:
        return credentials.Certificate("serviceAccountKey.json")
    else:
        return None

if not firebase_admin._apps:
    cred = get_firebase_credentials()
    if cred:
        firebase_admin.initialize_app(cred, {'storageBucket': BUCKET_NAME})
    else:
        st.error("❌ 인증 키 에러")
        st.stop()
        
db = firestore.client()
bucket = storage.bucket()

def get_drive_service():
    SCOPES = ['https://www.googleapis.com/auth/drive']
    if "firebase" in st.secrets:
        key_dict = dict(st.secrets["firebase"])
        creds = service_account.Credentials.from_service_account_info(key_dict, scopes=SCOPES)
    else:
        creds = service_account.Credentials.from_service_account_file("serviceAccountKey.json", scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

OPTIONS = {
    "subject": ["수학II", "수학I", "미적분", "확률과통계", "기하", "공통수학"],
    "grade": ["고2", "고1", "고3", "N수", "중등"],
    "unit_major": [
        "함수의 극한과 연속", "미분법", "적분법", "지수함수와 로그함수", "삼각함수", "수열",
        "순열과 조합", "확률", "통계", "이차곡선", "평면벡터", "공간도형", "다항식", "방정식", "행렬", "기타"
    ],
    "difficulty": ["상", "최상(Killer)", "중", "하", "최하"],
    "question_type": ["추론형", "계산형", "이해형", "문제해결형", "합답형"],
    "source_org": ["평가원", "교육청", "사관학교/경찰대", "EBS", "내신", "기타"],
    "concepts": ["샌드위치 정리", "절댓값 함수", "미분계수", "평균값 정리", "롤의 정리", "사이값 정리", "극대/극소", "변곡점", "정적분", "부분적분", "치환적분", "도함수 활용", "기타"] 
}

# ==========================================
# 2. Helper Functions
# ==========================================
def list_drive_images(folder_id):
    try:
        service = get_drive_service()
        query = f"'{folder_id}' in parents and (mimeType contains 'image/') and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        return results.get('files', [])
    except:
        return []

def download_image_from_drive(file_id):
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    file_obj = io.BytesIO()
    downloader = MediaIoBaseDownload(file_obj, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    file_obj.seek(0)
    img = Image.open(file_obj)
    # RGBA(투명) 이미지는 JPG 저장 시 에러나므로 RGB로 변환
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    return img

def move_file_to_done(file_id, current_folder_id, done_folder_id):
    try:
        service = get_drive_service()
        service.files().update(fileId=file_id, addParents=done_folder_id, removeParents=current_folder_id).execute()
        return True
    except:
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

def suggest_boxes_gemini(image, count):
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        return []

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        prompt = f"""
        Find exactly {count} math problems in this image.
        Return ONLY a JSON list of bounding boxes in [ymin, xmin, ymax, xmax] format (scale 0-1000).
        Example: [[0, 0, 500, 1000], [500, 0, 1000, 1000]]
        """
        response = model.generate_content([prompt, image])
        text = response.text.replace("```json", "").replace("```", "")
        return json.loads(text)
    except:
        return []

def analyze_cropped_image(image):
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = """
    Analyze this math problem.
    1. Convert content to LaTeX.
    2. Output JSON: {"problem_text": "...", "diagram_desc": "..."}
    """
    try:
        response = model.generate_content([prompt, image])
        text = response.text.replace("```json", "").replace("```", "")
        return json.loads(text)
    except:
        return {}

# ==========================================
# 3. Main UI
# ==========================================
st.title("📐 Math Labeling Studio (AI Assist)")

with st.sidebar:
    st.header("⚙️ 설정")
    default_folder = st.secrets.get("DEFAULT_FOLDER_ID", "")
    done_folder_default = st.secrets.get("DONE_FOLDER_ID", "")
    folder_id = st.text_input("작업 폴더 ID", value=default_folder)
    done_folder_id = st.text_input("완료 폴더 ID", value=done_folder_default)
    
    if st.button("📂 드라이브 스캔"):
        if folder_id:
            files = list_drive_images(folder_id)
            st.session_state['drive_files'] = files
            st.session_state['idx'] = 0

    st.markdown("---")
    c1, c2 = st.columns(2)
    if c1.button("◀ 이전"):
        if st.session_state.get('idx', 0) > 0:
            st.session_state['idx'] -= 1
            st.session_state.pop('canvas_init', None)
            st.session_state.pop('final_results', None)
            st.rerun()
            
    if c2.button("다음 ▶"):
        if st.session_state.get('drive_files') and st.session_state['idx'] < len(st.session_state['drive_files']) - 1:
            st.session_state['idx'] += 1
            st.session_state.pop('canvas_init', None)
            st.session_state.pop('final_results', None)
            st.rerun()

if 'drive_files' in st.session_state and st.session_state['drive_files']:
    files = st.session_state['drive_files']
    idx = st.session_state['idx']
    
    if idx >= len(files):
        st.info("완료!")
        st.stop()
        
    current_file = files[idx]
    
    # 이미지 로드
    if 'current_file_id' not in st.session_state or st.session_state['current_file_id'] != current_file['id']:
        st.session_state['original_img'] = download_image_from_drive(current_file['id'])
        st.session_state['current_file_id'] = current_file['id']
        st.session_state.pop('canvas_init', None)
        st.session_state.pop('final_results', None)

    original_img = st.session_state['original_img']
    img_w, img_h = original_img.size

    # [핵심 변경] 캔버스 표시용 리사이징 (너비 600px 고정)
    CANVAS_WIDTH = 600
    scale_factor = img_w / CANVAS_WIDTH
    canvas_height = int(img_h / scale_factor)
    
    # 캔버스용 이미지 생성 (Resized)
    resized_img = original_img.resize((CANVAS_WIDTH, canvas_height))

    # ==========================================
    # Step 1: AI 제안 및 캔버스 설정
    # ==========================================
    col_ctrl, col_canvas = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("1️⃣ 영역 설정")
        prob_count = st.number_input("문제 개수", min_value=1, max_value=10, value=2)
        
        # [AI 제안 버튼]
        if st.button("🤖 AI 영역 제안"):
            with st.spinner("AI가 문제 위치를 찾고 있습니다..."):
                boxes = suggest_boxes_gemini(original_img, prob_count)
                
                initial_objects = []
                for box in boxes:
                    # Gemini [ymin, xmin, ymax, xmax] (0-1000)
                    ymin, xmin, ymax, xmax = box
                    
                    # 1000분율 -> 캔버스 픽셀(600px 기준)로 변환
                    rect = {
                        "type": "rect",
                        "left": xmin / 1000 * CANVAS_WIDTH,
                        "top": ymin / 1000 * canvas_height,
                        "width": (xmax - xmin) / 1000 * CANVAS_WIDTH,
                        "height": (ymax - ymin) / 1000 * canvas_height,
                        "fill": "rgba(255, 165, 0, 0.3)",
                        "stroke": "#FF0000",
                        "strokeWidth": 2
                    }
                    initial_objects.append(rect)
                
                st.session_state['canvas_init'] = {
                    "version": "4.4.0",
                    "objects": initial_objects
                }
        
        st.info("👉 오른쪽에서 박스를 수정하세요.")
        
        # [최종 분석 버튼]
        if st.button("⚡ 자르기 및 분석", type="primary"):
            if 'canvas_result' in st.session_state and st.session_state['canvas_result'].json_data:
                objects = st.session_state['canvas_result'].json_data["objects"]
                
                if len(objects) == 0:
                    st.error("박스가 없습니다.")
                else:
                    results = []
                    with st.spinner(f"{len(objects)}개 문제 분석 중..."):
                        for i, obj in enumerate(objects):
                            # 캔버스 좌표 -> 원본 좌표 변환
                            left = int(obj["left"] * scale_factor)
                            top = int(obj["top"] * scale_factor)
                            width = int(obj["width"] * scale_factor)
                            height = int(obj["height"] * scale_factor)
                            
                            # 원본에서 자르기
                            crop_img = original_img.crop((left, top, left+width, top+height))
                            
                            # AI 분석
                            analysis = analyze_cropped_image(crop_img)
                            results.append({"img": crop_img, "data": analysis})
                    
                    st.session_state['final_results'] = results

    with col_canvas:
        # 캔버스 그리기 (배경은 리사이즈된 이미지 사용)
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_color="#FF0000",
            background_image=resized_img, # [변경] 원본 대신 리사이즈된
