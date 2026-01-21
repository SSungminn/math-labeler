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
from streamlit_cropper import st_cropper  # 이미지 자르는 도구

# ==========================================
# 0. 사용자 설정 (여기에 복사한 주소 넣기!)
# ==========================================
# 예: "math-problem-collector.appspot.com" (gs://는 빼고 넣으세요)
BUCKET_NAME = "math-problem-collector.firebasestorage.app" 

# ==========================================
# 1. Configuration & Auth
# ==========================================
st.set_page_config(layout="wide", page_title="Cloud Math Labeler")

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
        # Storage 사용을 위해 bucket 설정 추가
        firebase_admin.initialize_app(cred, {
            'storageBucket': BUCKET_NAME
        })
    else:
        st.error("❌ 인증 키를 찾을 수 없습니다.")
        st.stop()
        
db = firestore.client()
bucket = storage.bucket() # 스토리지 버킷 연결

def get_drive_service():
    if "firebase" in st.secrets:
        key_dict = dict(st.secrets["firebase"])
        creds = service_account.Credentials.from_service_account_info(
            key_dict, scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
    else:
        creds = service_account.Credentials.from_service_account_file(
            "serviceAccountKey.json", scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
    return build('drive', 'v3', credentials=creds)

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

# ==========================================
# 2. Helper Functions
# ==========================================

def list_drive_images(folder_id):
    try:
        service = get_drive_service()
        query = f"'{folder_id}' in parents and (mimeType contains 'image/') and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        return results.get('files', [])
    except Exception as e:
        st.error(f"드라이브 접근 오류: {e}")
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
    return Image.open(file_obj)

def upload_image_to_storage(image, filename):
    # 이미지를 바이트로 변환
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Firebase Storage에 업로드
    path = f"cropped_problems/{filename}"
    blob = bucket.blob(path)
    blob.upload_from_string(img_byte_arr, content_type='image/jpeg')
    blob.make_public() # 공개 URL 생성
    return blob.public_url

def extract_gemini(image):
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        return {"error": "API Key Missing"}

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # 이미지가 잘려서 들어오므로, 단일 문제로 인식하게 함
        prompt = """
        Analyze this math problem image.
        1. Convert equations to LaTeX ($...$).
        2. Output JSON: {"problem_text": "...", "diagram_desc": "..."}
        """
        response = model.generate_content([prompt, image])
        text = response.text
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 3. Main UI
# ==========================================
st.title("✂️ Cloud Math Cropper & Labeler")

with st.sidebar:
    st.header("⚙️ 설정")
    default_folder = st.secrets["DEFAULT_FOLDER_ID"] if "DEFAULT_FOLDER_ID" in st.secrets else ""
    folder_id = st.text_input("Drive Folder ID", value=default_folder)
    
    if st.button("📂 드라이브 불러오기"):
        if folder_id:
            with st.spinner("스캔 중..."):
                files = list_drive_images(folder_id)
                st.session_state['drive_files'] = files
                st.session_state['idx'] = 0
                st.success(f"{len(files)}개 파일 발견!")
    
    st.markdown("---")
    col_prev, col_next = st.columns(2)
    if col_prev.button("◀ 이전 파일"):
        if st.session_state.get('idx', 0) > 0:
            st.session_state['idx'] -= 1
            if 'cropped_img' in st.session_state: del st.session_state['cropped_img']
            if 'extracted' in st.session_state: del st.session_state['extracted']
            st.rerun()
            
    if col_next.button("다음 파일 ▶"):
        if 'drive_files' in st.session_state and st.session_state['idx'] < len(st.session_state['drive_files']) - 1:
            st.session_state['idx'] += 1
            if 'cropped_img' in st.session_state: del st.session_state['cropped_img']
            if 'extracted' in st.session_state: del st.session_state['extracted']
            st.rerun()

if 'drive_files' in st.session_state and st.session_state['drive_files']:
    files = st.session_state['drive_files']
    idx = st.session_state['idx']
    current_file = files[idx]
    
    st.subheader(f"🖼️ 원본: {current_file['name']}")
    
    # 1. 이미지 로드 및 크롭 도구 표시
    try:
        if 'original_img' not in st.session_state or st.session_state.get('current_file_id') != current_file['id']:
            st.session_state['original_img'] = download_image_from_drive(current_file['id'])
            st.session_state['current_file_id'] = current_file['id']
        
        # 크롭 UI
        st.info("마우스로 문제 영역을 드래그해서 선택하세요.")
        cropped_img = st_cropper(st.session_state['original_img'], realtime_update=True, box_color='#FF0000', aspect_ratio=None)
        
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.markdown("##### ✂️ 선택된 영역 미리보기")
            st.image(cropped_img, use_container_width=True)
            
        with col_c2:
            st.markdown("##### ⚡ AI 분석")
            if st.button("선택 영역 분석하기", type="primary"):
                with st.spinner("자른 이미지 분석 중..."):
                    st.session_state['cropped_img'] = cropped_img # 저장용으로 세션에 보관
                    extracted = extract_gemini(cropped_img)
                    if "error" in extracted:
                        st.error(extracted['error'])
                    else:
                        st.session_state['extracted'] = extracted
                        st.success("분석 완료!")

    except Exception as e:
        st.error(f"이미지 로드 실패: {e}")

    st.divider()

    # 2. 데이터 입력 및 저장
    if 'extracted' in st.session_state:
        item = st.session_state['extracted']
        
        with st.form("save_form"):
            st.subheader("📝 데이터 확인 및 저장")
            
            # [1열] 기본 정보
            c1, c2, c3, c4 = st.columns(4)
            subject = c1.selectbox("과목", OPTIONS['subject'])
            grade = c2.selectbox("학년", OPTIONS['grade'])
            source = c3.selectbox("출처", OPTIONS['source_org'])
            unit = c4.selectbox("단원", OPTIONS['unit_major'])
            
            # [2열] 심화 정보
            c5, c6, c7 = st.columns(3)
            diff = c5.selectbox("난이도", OPTIONS['difficulty'])
            q_type = c6.selectbox("유형", OPTIONS['question_type'])
            concept = c7.selectbox("핵심 개념", OPTIONS['concepts'])
            
            st.markdown("---")
            prob = st.text_area("문제 (LaTeX)", value=item.get('problem_text', ""), height=150)
            desc = st.text_area("도형 설명", value=item.get('diagram_desc', ""), height=80)
            
            if st.form_submit_button("🔥 이미지와 함께 저장"):
                if 'cropped_img' in st.session_state:
                    with st.spinner("이미지 업로드 및 DB 저장 중..."):
                        # 1. 이미지 이름 생성 (원본이름_시간.jpg)
                        timestamp = int(time.time())
                        clean_name = current_file['name'].rsplit('.', 1)[0]
                        img_filename = f"{clean_name}_{timestamp}.jpg"
                        
                        # 2. Storage에 업로드하고 URL 받기
                        img_url = upload_image_to_storage(st.session_state['cropped_img'], img_filename)
                        
                        # 3. Firestore에 데이터 저장 (이미지 URL 포함)
                        doc_data = {
                            "original_filename": current_file['name'],
                            "drive_file_id": current_file['id'],
                            "image_url": img_url,  # 핵심: 자른 이미지의 링크
                            "storage_path": f"cropped_problems/{img_filename}",
                            "meta": {
                                "subject": subject, "grade": grade, "source": source,
                                "unit": unit, "difficulty": diff, "question_type": q_type,
                                "concept": concept
                            },
                            "content": {"problem": prob, "diagram": desc},
                            "created_at": firestore.SERVER_TIMESTAMP
                        }
                        
                        db.collection("math_dataset").add(doc_data)
                        st.toast("저장 성공! 이미지가 클라우드에 안전하게 보관되었습니다.")
                        time.sleep(1)
                else:
                    st.error("분석된 이미지가 없습니다. 위에서 '선택 영역 분석하기'를 먼저 눌러주세요.")

else:
    st.info("왼쪽 사이드바에서 드라이브를 불러와주세요.")
