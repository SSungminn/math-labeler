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
import time

# ==========================================
# 1. Configuration & Auth
# ==========================================
st.set_page_config(layout="wide", page_title="Cloud Math Labeler")

# 파이어베이스 인증
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
        firebase_admin.initialize_app(cred)
    else:
        st.error("❌ 인증 키를 찾을 수 없습니다.")
        st.stop()
        
db = firestore.client()

# 구글 드라이브 API
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

def extract_gemini(image):
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        return [{"error": "API Key Missing"}]

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # 프롬프트 강화: 무조건 리스트 형식으로 반환하도록 강제
        prompt = """
        Analyze this math image. It may contain one or multiple problems.
        Extract each problem separately.
        
        Output format must be a JSON LIST of objects:
        [
            {
                "problem_text": "LaTeX code for problem 1...",
                "diagram_desc": "Description for problem 1..."
            },
            {
                "problem_text": "LaTeX code for problem 2...",
                "diagram_desc": "Description for problem 2..."
            }
        ]
        Do not include markdown format like ```json. Just raw JSON.
        """
        response = model.generate_content([prompt, image])
        text = response.text
        
        # 청소
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        result = json.loads(text)
        
        # 만약 AI가 리스트가 아니라 단일 객체를 줬다면 리스트로 감싸기 (방어 코드)
        if isinstance(result, dict):
            return [result]
        return result
        
    except Exception as e:
        return [{"error": str(e)}]

# ==========================================
# 3. Main UI
# ==========================================
st.title("☁️ Cloud Math Labeler (Multi-Problem Support)")

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
    # 네비게이션 버튼 (사이드바로 이동)
    col_prev, col_next = st.columns(2)
    if col_prev.button("◀ 이전"):
        if st.session_state.get('idx', 0) > 0:
            st.session_state['idx'] -= 1
            if 'extracted' in st.session_state: del st.session_state['extracted']
            st.rerun()
            
    if col_next.button("다음 ▶"):
        if 'drive_files' in st.session_state and st.session_state['idx'] < len(st.session_state['drive_files']) - 1:
            st.session_state['idx'] += 1
            if 'extracted' in st.session_state: del st.session_state['extracted']
            st.rerun()

if 'drive_files' in st.session_state and st.session_state['drive_files']:
    files = st.session_state['drive_files']
    idx = st.session_state['idx']
    current_file = files[idx]
    
    # 레이아웃: 위에는 이미지, 아래는 탭(Tab) 형식의 입력폼
    st.subheader(f"🖼️ ({idx+1}/{len(files)}) {current_file['name']}")
    
    # 1. 이미지 로드 및 AI 버튼
    col_img_view, col_action = st.columns([2, 1])
    with col_img_view:
        try:
            image = download_image_from_drive(current_file['id'])
            st.image(image, use_container_width=True)
        except Exception as e:
            st.error("이미지 로드 실패")

    with col_action:
        st.info("💡 이미지를 보고 AI 분석을 실행하세요.")
        if st.button("⚡ AI 자동 분석 (Extract)", type="primary"):
            with st.spinner("문제 추출 중..."):
                extracted_data = extract_gemini(image)
                # 결과가 에러인지 확인
                if isinstance(extracted_data, list) and "error" in extracted_data[0]:
                    st.error(extracted_data[0]["error"])
                else:
                    st.session_state['extracted'] = extracted_data
                    st.rerun()

    st.divider()

    # 2. 데이터 입력 영역 (탭으로 구분)
    if 'extracted' in st.session_state:
        data_list = st.session_state['extracted']
        
        # 탭 생성 (문제 개수만큼)
        tab_names = [f"문제 {i+1}" for i in range(len(data_list))]
        tabs = st.tabs(tab_names)
        
        for i, tab in enumerate(tabs):
            with tab:
                item = data_list[i]
                st.markdown(f"### 📝 문제 {i+1} 상세 입력")
                
                with st.form(f"form_{idx}_{i}"):
                    # [1열] 기본 정보 (과목, 학년, 출처, 단원)
                    c1, c2, c3, c4 = st.columns(4)
                    subject = c1.selectbox("과목", OPTIONS['subject'], key=f"sub_{idx}_{i}")
                    grade = c2.selectbox("학년", OPTIONS['grade'], key=f"grd_{idx}_{i}")
                    source = c3.selectbox("출처", OPTIONS['source_org'], key=f"src_{idx}_{i}")
                    unit = c4.selectbox("단원", OPTIONS['unit_major'], key=f"unt_{idx}_{i}")
                    
                    # [2열] 심화 정보 (난이도, 유형, 핵심개념)
                    c5, c6, c7 = st.columns(3)
                    diff = c5.selectbox("난이도", OPTIONS['difficulty'], key=f"dif_{idx}_{i}")
                    q_type = c6.selectbox("유형", OPTIONS['question_type'], key=f"typ_{idx}_{i}")
                    concept = c7.selectbox("핵심 개념", OPTIONS['concepts'], key=f"cpt_{idx}_{i}")
                    
                    st.markdown("---")
                    
                    # [텍스트] 문제 본문 & 설명
                    prob = st.text_area("문제 (LaTeX)", value=item.get('problem_text', ""), height=150, key=f"prb_{idx}_{i}")
                    desc = st.text_area("도형 설명", value=item.get('diagram_desc', ""), height=80, key=f"dsc_{idx}_{i}")
                    
                    # [저장 버튼]
                    if st.form_submit_button(f"💾 문제 {i+1} 저장"):
                        doc_data = {
                            "filename": current_file['name'],
                            "drive_file_id": current_file['id'],
                            "problem_index": i + 1,
                            "meta": {
                                "subject": subject, 
                                "grade": grade, 
                                "source": source,
                                "unit": unit, 
                                "difficulty": diff, 
                                "question_type": q_type,
                                "concept": concept  # 새로 추가된 항목
                            },
                            "content": {"problem": prob, "diagram": desc},
                            "created_at": firestore.SERVER_TIMESTAMP
                        }
                        
                        db.collection("math_dataset").add(doc_data)
                        st.success(f"✅ 문제 {i+1} 저장 완료!")
                        time.sleep(1)

else:
    st.info("왼쪽 사이드바에서 드라이브를 불러와주세요.")

