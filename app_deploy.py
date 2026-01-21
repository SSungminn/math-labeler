import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import json
import os
import google.generativeai as genai
from PIL import Image

# ==========================================
# 1. Configuration & Auth (Secure Way)
# ==========================================
st.set_page_config(layout="wide", page_title="Cloud Math Labeler")

# 파이어베이스 인증 함수
def get_firebase_credentials():
    # 1. Streamlit Secrets에 설정된 경우 (배포 환경)
    if "firebase" in st.secrets:
        return credentials.Certificate(dict(st.secrets["firebase"]))
    # 2. 로컬 파일이 있는 경우 (개발 환경)
    elif "serviceAccountKey.json" in [f.name for f in os.scandir('.')]:
        return credentials.Certificate("serviceAccountKey.json")
    else:
        return None

# A. Firebase 초기화 (Singleton)
if not firebase_admin._apps:
    cred = get_firebase_credentials()
    if cred:
        firebase_admin.initialize_app(cred)
    else:
        st.error("❌ 인증 키를 찾을 수 없습니다. Streamlit Secrets를 설정해주세요.")
        st.stop()
        
db = firestore.client()

# B. Google Drive API 연결
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

# 옵션 정의
OPTIONS = {
    "subject": ["수학II", "수학I", "미적분", "확률과통계", "기하", "공통수학"],
    "grade": ["고2", "고1", "고3", "N수", "중등"],
    "unit_major": ["함수의 극한과 연속", "미분", "적분", "지수/로그", "삼각함수", "수열"],
    "difficulty": ["상", "최상(Killer)", "중", "하", "최하"],
    "question_type": ["추론형", "계산형", "이해형", "문제해결형", "합답형"],
    "source_org": ["평가원", "교육청", "사관학교/경찰대", "EBS", "내신"]
}

# ==========================================
# 2. Helper Functions (Drive & AI)
# ==========================================

# 구글 드라이브 폴더에서 이미지 리스트 가져오기
def list_drive_images(folder_id):
    try:
        service = get_drive_service()
        query = f"'{folder_id}' in parents and (mimeType contains 'image/') and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        return results.get('files', [])
    except Exception as e:
        st.error(f"드라이브 접근 오류: {e}")
        return []

# 드라이브에서 이미지 다운로드
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

# Gemini AI 추출 (입력 인자에서 api_key 제거함 -> 내부에서 Secrets 사용)
def extract_gemini(image):
    # Secrets에서 안전하게 키 꺼내기
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        return {"error": "Secrets에 'GEMINI_API_KEY'가 설정되지 않았습니다."}

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = """
        수학 문제 이미지 분석:
        1. 수식은 LaTeX($...$)로 변환.
        2. JSON 포맷: {"problem_text": "...", "diagram_desc": "..."}
        """
        response = model.generate_content([prompt, image])
        text = response.text
        
        # JSON 파싱 (마크다운 제거)
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
st.title("☁️ Cloud Math Data Labeler")
st.caption("Storage: Firebase Firestore | Source: Google Drive")

with st.sidebar:
    st.header("⚙️ 설정")
    
    # [입력창 삭제됨] API Key 입력 부분 없음
    
    # 구글 드라이브 폴더 ID 입력
    # Secrets에 'DEFAULT_FOLDER_ID'가 있다면 기본값으로 사용
    default_folder = st.secrets["DEFAULT_FOLDER_ID"] if "DEFAULT_FOLDER_ID" in st.secrets else ""
    folder_id = st.text_input("Drive Folder ID", value=default_folder, placeholder="구글 드라이브 폴더 ID 붙여넣기")
    
    if st.button("📂 드라이브 불러오기"):
        if folder_id:
            with st.spinner("드라이브 스캔 중..."):
                files = list_drive_images(folder_id)
                st.session_state['drive_files'] = files
                st.session_state['idx'] = 0
                if files:
                    st.success(f"{len(files)}개 파일 발견!")
                else:
                    st.warning("이미지 파일이 없거나 폴더 ID가 잘못되었습니다.")
        else:
            st.error("폴더 ID를 입력하세요.")

if 'drive_files' in st.session_state and st.session_state['drive_files']:
    files = st.session_state['drive_files']
    idx = st.session_state['idx']
    
    # 끝까지 다 했는지 체크
    if idx >= len(files):
        st.balloons()
        st.success("🎉 모든 이미지 작업을 완료했습니다!")
        if st.button("처음으로 돌아가기"):
            st.session_state['idx'] = 0
            st.rerun()
        st.stop()
        
    current_file = files[idx]
    
    col1, col2 = st.columns([1, 1.2])
    
    # [왼쪽] 이미지 표시
    with col1:
        st.subheader(f"🖼️ ({idx+1}/{len(files)}) {current_file['name']}")
        try:
            image = download_image_from_drive(current_file['id'])
            st.image(image, use_container_width=True)
            
            # [수정됨] extract_gemini(image) -> 인자 1개만 전달
            if st.button("⚡ AI 분석", key="ai_btn"):
                with st.spinner("Gemini가 문제를 분석 중입니다..."):
                    extracted = extract_gemini(image)
                    if "error" in extracted:
                        st.error(f"오류: {extracted['error']}")
                    else:
                        st.session_state['extracted'] = extracted
                        st.success("분석 완료!")
        except Exception as e:
            st.error(f"이미지 로드 실패: {e}")

    # [오른쪽] 입력 폼
    with col2:
        st.subheader("📝 Firebase 저장")
        ai_data = st.session_state.get('extracted', {})
        
        with st.form("cloud_form"):
            c1, c2 = st.columns(2)
            subject = c1.selectbox("과목", OPTIONS['subject'])
            grade = c2.selectbox("학년", OPTIONS['grade'])
            
            c3, c4 = st.columns(2)
            unit = c3.text_input("단원", value="미분")
            diff = c4.selectbox("난이도", OPTIONS['difficulty'])
            
            prob = st.text_area("문제 (LaTeX)", value=ai_data.get('problem_text', ""), height=150)
            desc = st.text_area("도형 설명", value=ai_data.get('diagram_desc', ""), height=80)
            
            if st.form_submit_button("🔥 Firebase에 저장"):
                # Firestore 저장 로직
                doc_data = {
                    "filename": current_file['name'],
                    "drive_file_id": current_file['id'],
                    "meta": {"subject": subject, "grade": grade, "unit": unit, "difficulty": diff},
                    "content": {"problem": prob, "diagram": desc},
                    "created_at": firestore.SERVER_TIMESTAMP
                }
                
                # 컬렉션 이름: math_dataset
                db.collection("math_dataset").add(doc_data)
                
                st.toast("저장 완료! 다음 문제로...")
                time.sleep(0.5)
                st.session_state['idx'] += 1
                if 'extracted' in st.session_state:
                    del st.session_state['extracted']
                st.rerun()

else:
    st.info("왼쪽 사이드바에 'Drive Folder ID'를 넣고 불러오세요.")
    st.markdown("""
    **Tip:** 폴더 ID는 구글 드라이브 주소창에서 확인 가능합니다.
    `drive.google.com/drive/u/0/folders/` 뒤에 있는 **긴 문자열**입니다.
    """)
