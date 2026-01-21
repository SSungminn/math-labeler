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
st.set_page_config(layout="wide", page_title="Cloud Math Labeler AI+")

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
    구글 드라이브 인증. token_uri 누락 패치 포함.
    """
    SCOPES = ['https://www.googleapis.com/auth/drive']
    creds = None
    
    try:
        if "firebase" in st.secrets:
            key_dict = dict(st.secrets["firebase"])
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

# 분류 옵션 정의
OPTIONS = {
    "subject": ["수학II", "수학I", "미적분", "확률과통계", "기하", "공통수학", "중등수학"],
    "grade": ["고2", "고1", "고3", "N수", "중등"],
    "unit_major": [
        "함수의 극한과 연속", "미분법", "적분법", 
        "지수함수와 로그함수", "삼각함수", "수열",
        "순열과 조합", "확률", "통계",
        "이차곡선", "평면벡터", "공간도형과 공간좌표",
        "다항식", "방정식과 부등식", "행렬", "집합과 명제", "함수", "기타"
    ],
    "difficulty": ["상", "최상(Killer)", "중", "하", "최하"],
    "question_type": ["추론형", "계산형", "이해형", "문제해결형", "합답형"],
    "source_org": ["평가원", "교육청", "사관학교/경찰대", "EBS", "내신", "기타"],
    "concepts": ["샌드위치 정리", "절댓값 함수", "미분계수의 정의", "평균값 정리", "롤의 정리", "사이값 정리", "극대/극소", "변곡점", "정적분 정의", "부분적분", "치환적분", "도함수 활용", "삼수선의 정리", "기타"] 
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

def extract_gemini(image, options_dict):
    """
    이미지를 분석하여 텍스트, 도형 설명 및 카테고리 분류를 수행합니다.
    options_dict를 프롬프트에 포함시켜 AI가 선택지 내에서 답을 고르도록 유도합니다.
    """
    if "GEMINI_API_KEY" not in st.secrets:
        return {"error": "API Key Missing in Secrets"}

    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-2.0-flash") 
        
        # 옵션 리스트를 문자열로 변환하여 프롬프트에 주입
        options_str = json.dumps(options_dict, ensure_ascii=False, indent=2)

        prompt = f"""
        당신은 한국의 고등학교 수학 전문가입니다. 이 수학 문제 이미지를 완벽하게 분석하세요.
        
        [지시사항]
        1. **수식 추출**: 모든 수식은 LaTeX 포맷($...$)으로 변환하세요.
        2. **문제 텍스트**: 문제의 지문 내용을 한국어 그대로 추출하세요.
        3. **도형 설명**: 도형이나 그래프가 있다면 'diagram_desc'에 한국어로 자세히 묘사하세요.
        4. **자동 분류**: 아래 제공된 [분류 리스트]를 참고하여, 이 문제에 가장 적합한 항목을 하나씩 선택하세요.
           (반드시 리스트 안에 있는 단어만 사용해야 합니다.)

        [분류 리스트]
        {options_str}

        [출력 포맷]
        반드시 아래의 JSON 형식으로만 출력하세요 (마크다운 없이 순수 JSON):
        {{
            "problem_text": "추출된 문제 내용...",
            "diagram_desc": "도형 설명...",
            "subject": "분류 리스트의 subject 중 택1",
            "unit_major": "분류 리스트의 unit_major 중 택1",
            "question_type": "분류 리스트의 question_type 중 택1",
            "concept": "분류 리스트의 concepts 중 택1 (없으면 '기타')",
            "difficulty": "분류 리스트의 difficulty 중 택1 (추정)"
        }}
        """
        
        response = model.generate_content([prompt, image])
        text = response.text
        
        # JSON 파싱
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            clean_json = json_match.group(0)
            return json.loads(clean_json)
        else:
            return {"problem_text": text, "diagram_desc": "JSON 파싱 실패", "error": "Format Error"}
            
    except Exception as e:
        return {"error": str(e)}

def get_index_or_default(options_list, value, default_index=0):
    """AI가 예측한 값이 리스트에 있으면 그 인덱스를 반환, 없으면 0 반환"""
    try:
        return options_list.index(value)
    except ValueError:
        return default_index

# ==========================================
# 3. 메인 UI 레이아웃
# ==========================================
st.title("✂️ Smart Math Labeler (AI Classification)")

with st.sidebar:
    st.header("⚙️ 설정")
    
    default_folder = st.secrets.get("DEFAULT_FOLDER_ID", "")
    done_folder_default = st.secrets.get("DONE_FOLDER_ID", "")
    
    folder_id = st.text_input("작업 폴더 ID (Source)", value=default_folder)
    done_folder_id = st.text_input("완료 폴더 ID (Done)", value=done_folder_default)
    
    if st.button("📂 드라이브 불러오기", type="primary"):
        if folder_id:
            with st.spinner("파일 스캔 중..."):
                files = list_drive_images(folder_id)
                st.session_state['drive_files'] = files
                st.session_state['idx'] = 0
                st.session_state.pop('cropped_img', None)
                st.session_state.pop('extracted', None)
                st.success(f"{len(files)}개 이미지 발견!")
        else:
            st.warning("폴더 ID를 입력하세요.")

    st.markdown("---")
    
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
    
    if idx >= len(files):
        st.session_state['idx'] = 0
        st.rerun()
        
    current_file = files[idx]
    st.subheader(f"🖼️ [{idx+1}/{len(files)}] {current_file['name']}")
    
    if 'current_file_id' not in st.session_state or st.session_state['current_file_id'] != current_file['id']:
        with st.spinner("이미지 로딩 중..."):
            img = download_image_from_drive(current_file['id'])
            if img:
                st.session_state['original_img'] = img
                st.session_state['current_file_id'] = current_file['id']
                st.session_state.pop('cropped_img', None)
                st.session_state.pop('extracted', None)
    
    if 'original_img' in st.session_state:
        st.info("💡 문제 영역을 드래그하세요.")
        cropped_img = st_cropper(
            st.session_state['original_img'],
            realtime_update=True,
            box_color='#FF0000',
            aspect_ratio=None
        )
        
        col_view, col_action = st.columns([1, 1])
        with col_view:
            st.image(cropped_img, use_container_width=True)
        with col_action:
            if st.button("✨ AI 분석 및 자동 분류", type="primary"):
                with st.spinner("Gemini가 문제를 풀고 분류 중입니다..."):
                    st.session_state['cropped_img'] = cropped_img
                    # 옵션 전체를 전달하여 AI가 판단하게 함
                    extracted_data = extract_gemini(cropped_img, OPTIONS)
                    
                    if "error" in extracted_data:
                        st.error(extracted_data['error'])
                    else:
                        st.session_state['extracted'] = extracted_data
                        st.success("분석 완료!")

    st.divider()

    if 'extracted' in st.session_state:
        item = st.session_state['extracted']
        
        # AI 예측값 가져오기 (없으면 기본값)
        pred_subject = item.get("subject", OPTIONS['subject'][0])
        pred_unit = item.get("unit_major", OPTIONS['unit_major'][0])
        pred_type = item.get("question_type", OPTIONS['question_type'][0])
        pred_concept = item.get("concept", OPTIONS['concepts'][-1]) # 기본값 기타
        pred_diff = item.get("difficulty", "중")

        with st.form("labeling_form"):
            st.subheader("📝 AI 자동 분류 결과 확인")
            
            # AI가 예측한 인덱스를 기본값으로 설정
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            subject = r1c1.selectbox("과목", OPTIONS['subject'], index=get_index_or_default(OPTIONS['subject'], pred_subject))
            grade = r1c2.selectbox("학년", OPTIONS['grade'], index=0) # 학년은 이미지로 알기 어려움
            source = r1c3.selectbox("출처", OPTIONS['source_org'], index=0) # 출처도 알기 어려움
            unit = r1c4.selectbox("단원", OPTIONS['unit_major'], index=get_index_or_default(OPTIONS['unit_major'], pred_unit))
            
            r2c1, r2c2, r2c3 = st.columns(3)
            diff = r2c1.selectbox("난이도", OPTIONS['difficulty'], index=get_index_or_default(OPTIONS['difficulty'], pred_diff))
            q_type = r2c2.selectbox("유형", OPTIONS['question_type'], index=get_index_or_default(OPTIONS['question_type'], pred_type))
            concept = r2c3.selectbox("핵심 개념", OPTIONS['concepts'], index=get_index_or_default(OPTIONS['concepts'], pred_concept))
            
            st.markdown("---")
            prob_text = st.text_area("문제 (LaTeX)", value=item.get('problem_text', ""), height=200)
            diag_desc = st.text_area("도형 설명", value=item.get('diagram_desc', ""), height=100)
            
            if st.form_submit_button("🔥 저장 및 파일 이동"):
                if 'cropped_img' not in st.session_state:
                    st.error("이미지 없음")
                else:
                    try:
                        with st.spinner("업로드 및 저장 중..."):
                            timestamp = int(time.time())
                            clean_name = re.sub(r'[^a-zA-Z0-9가-힣_-]', '', current_file['name'].rsplit('.', 1)[0])
                            img_filename = f"{clean_name}_{timestamp}.jpg"
                            img_url = upload_image_to_storage(st.session_state['cropped_img'], img_filename)
                        
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
                                "labeler_version": "v3.0-ai-auto-class"
                            }
                            db.collection("math_dataset").add(doc_data)
                            
                        if done_folder_id:
                            move_file_to_done(current_file['id'], folder_id, done_folder_id)
                            st.toast("✅ 저장 완료!")
                            st.session_state['drive_files'].pop(idx)
                            st.session_state.pop('cropped_img', None)
                            st.session_state.pop('extracted', None)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.warning("저장됨 (파일 이동 안함)")
                    except Exception as e:
                        st.error(f"Error: {e}")

else:
    st.info("👈 드라이브 연결 필요")
