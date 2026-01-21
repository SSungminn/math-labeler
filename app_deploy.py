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
    if "GEMINI_API_KEY" not in st.secrets:
        return {"error": "API Key Missing"}

    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-2.0-flash") 
        
        options_str = json.dumps(options_dict, ensure_ascii=False, indent=2)

        # [중요] diagram_code 요청이 포함된 프롬프트
        prompt = f"""
        당신은 한국의 고등학교 수학 전문가이자 Python 시각화 전문가입니다.
        
        [지시사항]
        1. **수식 추출**: LaTeX 포맷($...$)으로 변환.
        2. **문제 텍스트**: 한국어 그대로 추출.
        3. **도형 코드 생성(핵심)**: 
           - 이미지의 도형/그래프를 Python `matplotlib`로 그리는 **실행 가능한 코드**를 작성하세요.
           - `import matplotlib.pyplot as plt` 필수.
           - 결과 객체는 반드시 `fig` 변수에 할당. (예: `fig, ax = plt.subplots()`)
           - 한글 폰트 설정 제외 (시스템 기본 사용).
           - 코드는 JSON의 "diagram_code" 필드에 문자열로 넣으세요.
        4. **자동 분류**: 아래 리스트 참고.

        [분류 리스트]
        {options_str}

        [출력 포맷 (JSON)]
        {{
            "problem_text": "...",
            "diagram_code": "import matplotlib.pyplot as plt\\n...",
            "diagram_desc": "...",
            "subject": "...",
            "unit_major": "...",
            "question_type": "...",
            "concept": "...",
            "difficulty": "..."
        }}
        """
        
        response = model.generate_content([prompt, image])
        text = response.text
        
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            return {"problem_text": text, "error": "Format Error"}
            
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

    # ... (이전 코드: st.cropper 등) ...

    if 'extracted' in st.session_state:
        item = st.session_state['extracted']
        
        # 기본값 로드
        default_prob = item.get('problem_text', "")
        default_code = item.get('diagram_code', "")
        
        st.divider()
        st.subheader("📝 데이터 검증 및 저장")

        # [변경] st.form을 제거하여 실시간 인터랙션 허용
        # 1. 메타데이터 선택 (즉시 반영되어도 상관없음)
        c1, c2, c3, c4 = st.columns(4)
        subject = c1.selectbox("과목", OPTIONS['subject'], index=get_index_or_default(OPTIONS['subject'], item.get("subject")))
        grade = c2.selectbox("학년", OPTIONS['grade'], index=0)
        source = c3.selectbox("출처", OPTIONS['source_org'], index=0)
        unit = c4.selectbox("단원", OPTIONS['unit_major'], index=get_index_or_default(OPTIONS['unit_major'], item.get("unit_major")))
        
        c5, c6, c7 = st.columns(3)
        diff = c5.selectbox("난이도", OPTIONS['difficulty'], index=get_index_or_default(OPTIONS['difficulty'], item.get("difficulty")))
        q_type = c6.selectbox("유형", OPTIONS['question_type'], index=get_index_or_default(OPTIONS['question_type'], item.get("question_type")))
        concept = c7.selectbox("핵심 개념", OPTIONS['concepts'], index=get_index_or_default(OPTIONS['concepts'], item.get("concept")))

        st.markdown("---")

        # 2. 실시간 편집 & 미리보기 (Editor & Preview)
        col_edit, col_preview = st.columns(2)
        
        with col_edit:
            st.markdown("#### ✏️ 편집기")
            # 문제 텍스트 수정
            prob_text = st.text_area("문제 (LaTeX)", value=default_prob, height=300, key="prob_input")
            
            # 그래프 코드 수정
            st.caption("도형 Python 코드")
            diag_code = st.text_area("Matplotlib Code", value=default_code, height=200, key="code_input")
            
            # 도형 설명 텍스트
            diag_desc = st.text_area("도형 설명 (텍스트)", value=item.get('diagram_desc', ""), height=100)

        with col_preview:
            st.markdown("#### 👁️ 미리보기")
            
            # (A) 텍스트 렌더링
            if prob_text:
                st.info("수식 렌더링 확인")
                st.markdown(prob_text)
            else:
                st.warning("텍스트가 없습니다.")
            
            # (B) 그래프 렌더링 (자동 실행)
            if diag_code and "plt" in diag_code:
                st.markdown("---")
                st.info("📊 그래프 렌더링 확인")
                try:
                    local_vars = {}
                    # exec는 안전하지 않지만, 내부 도구이므로 허용
                    exec(diag_code, globals(), local_vars)
                    if 'fig' in local_vars:
                        st.pyplot(local_vars['fig'])
                    else:
                        st.warning("코드는 실행됐으나 'fig' 변수가 없습니다.")
                except Exception as e:
                    st.error(f"그래프 오류: {e}")

        st.markdown("---")
        
        # 3. 최종 저장 버튼 (이것만 버튼으로 처리)
        # form이 없으므로 모든 변수(prob_text, diag_code 등)는 현재 상태값을 그대로 가져감
        if st.button("🔥 저장 및 파일 이동 (Save & Move)", type="primary", use_container_width=True):
            if 'cropped_img' not in st.session_state:
                st.error("이미지 세션이 만료되었습니다.")
            else:
                try:
                    with st.spinner("데이터 저장 중..."):
                        # 이미지 업로드
                        timestamp = int(time.time())
                        clean_name = re.sub(r'[^a-zA-Z0-9가-힣_-]', '', current_file['name'].rsplit('.', 1)[0])
                        img_filename = f"{clean_name}_{timestamp}.jpg"
                        img_url = upload_image_to_storage(st.session_state['cropped_img'], img_filename)
                        
                        # Firestore 저장
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
                            # 코드 데이터도 같이 저장
                            "content": {
                                "problem": prob_text, 
                                "diagram_desc": diag_desc,
                                "diagram_code": diag_code  # 코드 저장
                            },
                            "created_at": firestore.SERVER_TIMESTAMP,
                            "labeler_version": "v3.1-live-preview"
                        }
                        db.collection("math_dataset").add(doc_data)
                        
                        # 파일 이동
                        if done_folder_id:
                            move_file_to_done(current_file['id'], folder_id, done_folder_id)
                            st.toast("✅ 저장 완료!")
                            st.session_state['drive_files'].pop(idx)
                            st.session_state.pop('cropped_img', None)
                            st.session_state.pop('extracted', None)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.success("저장 완료 (파일 이동 안 함)")
                            
                except Exception as e:
                    st.error(f"저장 실패: {e}")

else:
    st.info("👈 드라이브 연결 필요")




