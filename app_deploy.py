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
# 0. 전역 상수 설정
# ==========================================
BUCKET_NAME = "math-problem-collector.firebasestorage.app"
TEMP_DIR = "temp_images"

# 임시 디렉토리 생성 (필요시)
os.makedirs(TEMP_DIR, exist_ok=True)

# ==========================================
# 1. 설정 및 인증 (Configuration & Auth)
# ==========================================
st.set_page_config(layout="wide", page_title="Cloud Math Labeler")

@st.cache_resource
def init_firebase():
    """
    Firebase 인증을 세션당 한 번만 수행하여 리소스 낭비를 막습니다.
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
        st.error(f"🔥 Firebase 초기화 오류: {e}")
        return None, None

@st.cache_resource
def get_drive_service():
    """
    구글 드라이브 인증을 수행하고 서비스 객체를 캐싱합니다.
    token_uri 누락으로 인한 'No access token' 오류를 방지하는 패치가 포함되어 있습니다.
    """
    SCOPES = ['https://www.googleapis.com/auth/drive']
    creds = None
    
    try:
        if "firebase" in st.secrets:
            # 1. secrets를 딕셔너리로 변환
            key_dict = dict(st.secrets["firebase"])
            
            # 2. [중요] token_uri가 없다면 강제로 주입 (이게 없으면 인증이 깨짐)
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
        st.error(f"🚗 드라이브 인증 오류: {e}")
        return None

# 리소스 초기화
db, bucket = init_firebase()
drive_service = get_drive_service()

if not db or not drive_service:
    st.error("❌ 치명적 오류: 인증 키를 찾을 수 없거나 올바르지 않습니다.")
    st.stop()

# ==========================================
# 2. 로직 및 데이터 처리
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
        st.error(f"파일 목록 조회 실패: {e}")
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
        st.error(f"이미지 다운로드 실패: {e}")
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
        st.error(f"파일 이동 실패: {e}")
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
        # 속도와 비용 효율성을 위해 flash 모델 권장
        model = genai.GenerativeModel("gemini-2.0-flash") 
        
        # [수정됨] 프롬프트를 한글로 변경하여 한국어 출력을 강제함
        prompt = """
        이 수학 문제 이미지를 분석하세요.
        1. 수식은 LaTeX 포맷($...$)으로 변환하고, 문제 텍스트는 이미지에 있는 그대로(한국어 포함) 추출하세요.
        2. 도형이나 그래프에 대한 설명(diagram_desc)은 반드시 '한국어'로 자세히 묘사하세요.
        3. 결과는 반드시 다음 키를 가진 JSON 객체로만 반환하세요: "problem_text", "diagram_desc".
        """
        
        response = model.generate_content([prompt, image])
        text = response.text
        
        # 견고한 JSON 파싱 (Regex 사용)
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            clean_json = json_match.group(0)
            return json.loads(clean_json)
        else:
            # JSON 파싱 실패 시 원문 반환
            return {"problem_text": text, "diagram_desc": "자동 추출 형식이 올바르지 않음."}
            
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 3. 메인 UI 레이아웃
# ==========================================
st.title("✂️ Cloud Math Cropper & Labeler")

with st.sidebar:
    st.header("⚙️ 설정")
    
    default_folder = st.secrets.get("DEFAULT_FOLDER_ID", "")
    done_folder_default = st.secrets.get("DONE_FOLDER_ID", "")
    
    folder_id = st.text_input("작업 폴더 ID (Source)", value=default_folder)
    done_folder_id = st.text_input("완료 폴더 ID (Done)", value=done_folder_default, placeholder="처리 후 이동할 폴더 ID")
    
    if st.button("📂 드라이브 불러오기", type="primary"):
        if folder_id:
            with st.spinner("파일 스캔 중..."):
                files = list_drive_images(folder_id)
                st.session_state['drive_files'] = files
                st.session_state['idx'] = 0
                # 이전 상태 초기화
                st.session_state.pop('cropped_img', None)
                st.session_state.pop('extracted', None)
                st.success(f"{len(files)}개 이미지 발견!")
        else:
            st.warning("작업 폴더 ID를 입력해주세요.")

    st.markdown("---")
    
    # 네비게이션
    c_prev, c_next = st.columns(2)
    with c_prev:
        if st.button("◀ 이전"):
            if st.session_state.get('idx', 0) > 0:
                st.session_state['idx'] -= 1
                st.session_state.pop('cropped_img', None)
                st.session_state.pop('extracted', None)
                st.rerun()
                
    with c_next:
        if st.button("다음 ▶"):
            files = st.session_state.get('drive_files', [])
            if files and st.session_state['idx'] < len(files) - 1:
                st.session_state['idx'] += 1
                st.session_state.pop('cropped_img', None)
                st.session_state.pop('extracted', None)
                st.rerun()

# ==========================================
# 4. 작업 공간
# ==========================================
if 'drive_files' in st.session_state and st.session_state['drive_files']:
    files = st.session_state['drive_files']
    idx = st.session_state['idx']
    
    # 인덱스 범위 안전장치 (파일 이동 후 리스트 변경 시 에러 방지)
    if idx >= len(files):
        st.warning("파일 목록이 변경되었습니다. 인덱스를 초기화합니다.")
        st.session_state['idx'] = 0
        st.rerun()
        
    current_file = files[idx]
    
    st.subheader(f"🖼️ [{idx+1}/{len(files)}] {current_file['name']}")
    
    # 이미지 로드 (세션 상태에 캐싱하여 불필요한 재다운로드 방지)
    if 'current_file_id' not in st.session_state or st.session_state['current_file_id'] != current_file['id']:
        with st.spinner("이미지 다운로드 중..."):
            img = download_image_from_drive(current_file['id'])
            if img:
                st.session_state['original_img'] = img
                st.session_state['current_file_id'] = current_file['id']
                # 새 이미지 로드 시 하위 상태 초기화
                st.session_state.pop('cropped_img', None)
                st.session_state.pop('extracted', None)
            else:
                st.error("이미지를 불러오지 못했습니다. 다음 파일로 넘어가주세요.")
    
    if 'original_img' in st.session_state:
        # 크롭 도구
        st.info("💡 마우스로 문제 영역을 드래그해서 선택하세요.")
        
        cropped_img = st_cropper(
            st.session_state['original_img'],
            realtime_update=True,
            box_color='#FF0000',
            aspect_ratio=None
        )
        
        col_view, col_action = st.columns([1, 1])
        
        with col_view:
            st.markdown("##### ✂️ 선택된 영역 미리보기")
            st.image(cropped_img, use_container_width=True)
            
        with col_action:
            st.markdown("##### ⚡ AI 분석")
            if st.button("✨ 선택 영역 분석하기", type="primary"):
                with st.spinner("Gemini가 분석 중입니다..."):
                    st.session_state['cropped_img'] = cropped_img
                    extracted_data = extract_gemini(cropped_img)
                    
                    if "error" in extracted_data:
                        st.error(extracted_data['error'])
                    else:
                        st.session_state['extracted'] = extracted_data
                        st.success("분석 완료!")

    st.divider()

    # 데이터 확인 및 저장 폼
    if 'extracted' in st.session_state:
        item = st.session_state['extracted']
        
        with st.form("labeling_form"):
            st.subheader("📝 데이터 검증 및 저장")
            
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            subject = r1c1.selectbox("과목", OPTIONS['subject'])
            grade = r1c2.selectbox("학년", OPTIONS['grade'])
            source = r1c3.selectbox("출처", OPTIONS['source_org'])
            unit = r1c4.selectbox("단원", OPTIONS['unit_major'])
            
            r2c1, r2c2, r2c3 = st.columns(3)
            diff = r2c1.selectbox("난이도", OPTIONS['difficulty'])
            q_type = r2c2.selectbox("유형", OPTIONS['question_type'])
            concept = r2c3.selectbox("핵심 개념", OPTIONS['concepts'])
            
            st.markdown("---")
            prob_text = st.text_area("문제 (LaTeX)", value=item.get('problem_text', ""), height=200)
            diag_desc = st.text_area("도형 설명", value=item.get('diagram_desc', ""), height=100)
            
            submit_btn = st.form_submit_button("🔥 저장 및 이동 (Save & Move)")
            
            if submit_btn:
                if 'cropped_img' not in st.session_state:
                    st.error("자른 이미지가 없습니다. 다시 선택해주세요.")
                else:
                    try:
                        # 1. 스토리지 업로드
                        with st.spinner("1. 이미지 업로드 중..."):
                            timestamp = int(time.time())
                            clean_name = current_file['name'].rsplit('.', 1)[0]
                            # 파일명 정제 (특수문자 제거)
                            clean_name = re.sub(r'[^a-zA-Z0-9가-힣_-]', '', clean_name)
                            img_filename = f"{clean_name}_{timestamp}.jpg"
                            
                            img_url = upload_image_to_storage(st.session_state['cropped_img'], img_filename)
                        
                        # 2. 메타데이터 저장
                        with st.spinner("2. 데이터베이스 저장 중..."):
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
                                "labeler_version": "v2.0-korean-optimized"
                            }
                            db.collection("math_dataset").add(doc_data)
                            
                        # 3. 파일 이동
                        if done_folder_id:
                            with st.spinner("3. 완료 폴더로 이동 중..."):
                                success = move_file_to_done(current_file['id'], folder_id, done_folder_id)
                                if success:
                                    st.toast("✅ 저장 및 파일 이동 완료!")
                                    # 로컬 리스트 업데이트 (인덱스 유지하면서 항목 제거)
                                    st.session_state['drive_files'].pop(idx)
                                    # 상태 정리
                                    st.session_state.pop('cropped_img', None)
                                    st.session_state.pop('extracted', None)
                                    # 리스트가 줄어들었으므로 인덱스 증가 없이 리로드
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("저장은 완료되었으나 드라이브 파일 이동에 실패했습니다.")
                        else:
                            st.warning("저장은 완료되었으나 '완료 폴더 ID'가 없어 파일 이동은 하지 않았습니다.")
                            
                    except Exception as e:
                        st.error(f"저장 실패: {e}")

else:
    st.info("👈 왼쪽 사이드바에서 드라이브를 연결해주세요.")
