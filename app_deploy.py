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
    Matplotlib 코드 생성을 강제하도록 프롬프트가 강화되었습니다.
    """
    if "GEMINI_API_KEY" not in st.secrets:
        return {"error": "API Key Missing in Secrets"}

    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        
        # JSON 모드 유지
        generation_config = {
            "temperature": 0.1, 
            "response_mime_type": "application/json"
        }
        
        # 2.0 Flash가 코드를 잘 못 짜면 1.5 Pro로 변경 고려 (일단 Flash 유지)
        model = genai.GenerativeModel("gemini-2.0-flash", generation_config=generation_config) 
        
        options_str = json.dumps(options_dict, ensure_ascii=False, indent=2)

        # [변경점] 프롬프트 강화: 조건부 삭제, 무조건 생성 지시, 예외 처리 추가
        prompt = f"""
        당신은 한국의 수학 교육 전문가이자 Python Matplotlib 코딩 전문가입니다.
        제공된 수학 문제 이미지를 분석하여 다음 작업을 수행하고 JSON으로 반환하세요.

        [작업 1: 텍스트 및 수식 추출]
        - 문제의 모든 텍스트를 한국어 그대로 추출하세요.
        - 수식은 LaTeX 포맷($...$)을 사용하세요.
        - 보기가 있다면 보기까지 모두 포함하세요.

        [작업 2: 도형 설명 (diagram_desc)]
        - 시각장애인을 위해 그래프나 도형의 생김새를 한국어로 상세히 묘사하세요.
        - "그림 참고"라고 하지 말고, "원점을 지나고 우상향하는 직선" 처럼 구체적으로 쓰세요.

        [작업 3: Python Matplotlib 코드 생성 (diagram_code)]
        - **매우 중요:** 이 필드는 절대로 비워둘 수 없습니다. ("" 금지)
        - 문제에 그래프, 도형, 함수식이 조금이라도 보인다면 그것을 그리는 코드를 작성하세요.
        - **만약 그림이 없는 단순 계산 문제라면, 빈 좌표평면(x축, y축)이라도 그리는 코드를 반드시 넣으세요.**
        - 코드는 반드시 `import matplotlib.pyplot as plt`와 `fig, ax = plt.subplots()`를 포함해야 합니다.
        - 한글 폰트 문제는 피하기 위해 라벨은 영어나 수식($...$)만 사용하세요.
        - JSON 문자열 안에 코드를 넣어야 하므로, 줄바꿈은 반드시 `\\n` 문자로 이스케이프 처리하여 한 줄로 작성되어야 합니다.

        [작업 4: 자동 분류]
        - 아래 리스트에서 가장 적절한 값을 선택하세요:
        {options_str}

        [응답 스키마 (JSON)]
        {{
            "problem_text": "추출된 텍스트...",
            "diagram_desc": "그래프 설명...",
            "diagram_code": "import matplotlib.pyplot as plt\\nfig, ax = plt.subplots()\\n...",
            "subject": "...",
            "unit_major": "...",
            "question_type": "...",
            "concept": "...",
            "difficulty": "..."
        }}
        """
        
        response = model.generate_content([prompt, image])
        text = response.text
        
        # Markdown 제거
        clean_text = re.sub(r"```json|```", "", text).strip()
            
        return json.loads(clean_text)
            
    except Exception as e:
        # 에러 발생 시에도 빈 값 대신 기본 템플릿 반환
        return {
            "error": f"Extraction Failed: {str(e)}", 
            "problem_text": "", 
            "diagram_code": "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.text(0.5, 0.5, 'Error generating graph', ha='center')"
        }
    try:
        # ... (모델 생성 및 프롬프트 코드 생략) ...
        
        response = model.generate_content([prompt, image])
        text = response.text
        
        # [디버깅 1] 터미널(콘솔)에 강제로 찍어서 확인 (서버 로그용)
        print("================ GEMINI RAW RESPONSE ================")
        print(text)
        print("=====================================================")

        # 혹시 모를 마크다운 태그 제거
        clean_text = re.sub(r"```json|```", "", text).strip()
            
        return json.loads(clean_text)
            
    except Exception as e:
        # [디버깅 2] 에러가 나면 'raw_text' 키에 원본을 담아서 리턴
        st.error(f"JSON 파싱 실패! 원본을 확인하세요.")
        return {
            "error": f"Extraction Failed: {str(e)}", 
            "problem_text": "", 
            "diagram_code": "",
            "raw_text_debug": text if 'text' in locals() else "No text generated" # 원본 텍스트 반환
        }

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
            st.image(cropped_img, width="stretch")
        with col_action:
            if st.button("✨ AI 분석 및 자동 분류", type="primary"):
                with st.spinner("Gemini가 문제를 풀고 분류 중입니다..."):
                    st.session_state['cropped_img'] = cropped_img
                    
                    extracted_data = extract_gemini(cropped_img, OPTIONS)
                    
                    # [디버깅 3] 화면에 디버그용 확장 패널 추가
                    with st.expander("🕵️‍♂️ AI 응답 데이터 뜯어보기 (Debug)", expanded=True):
                        st.json(extracted_data) # JSON 구조를 예쁘게 보여줌
                        
                        # 만약 에러가 나서 raw_text_debug가 있다면 원본 텍스트 출력
                        if "raw_text_debug" in extracted_data:
                            st.warning("⚠️ 파싱 실패한 원본 텍스트:")
                            st.code(extracted_data["raw_text_debug"], language="text")

                    if "error" in extracted_data:
                        st.error(extracted_data['error'])
                    else:
                        st.session_state['extracted'] = extracted_data
                        st.success("분석 완료!")

    st.divider()

    # ... (이전 코드: st.cropper 등) ...

    if 'extracted' in st.session_state:
        item = st.session_state['extracted']
        
        # 기본값 로드 (수정 중인 데이터가 날아가지 않게 session state 활용)
        default_prob = item.get('problem_text', "")
        default_code = item.get('diagram_code', "")
        
        st.divider()
        st.subheader("📝 데이터 검증 및 저장")

        # 1. 메타데이터 선택 (즉시 반영)
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

        # 2. 실시간 편집 & 미리보기
        col_edit, col_preview = st.columns(2)
        
        with col_edit:
            st.markdown("#### ✏️ 편집기")
            prob_text = st.text_area("문제 (LaTeX)", value=default_prob, height=300, key="prob_input")
            st.caption("도형 Python 코드")
            diag_code = st.text_area("Matplotlib Code", value=default_code, height=200, key="code_input")
            diag_desc = st.text_area("도형 설명 (텍스트)", value=item.get('diagram_desc', ""), height=100)

        with col_preview:
            st.markdown("#### 👁️ 미리보기")
            if prob_text:
                st.info("수식 렌더링 확인")
                st.markdown(prob_text)
            else:
                st.warning("텍스트가 없습니다.")
            
            if diag_code and "plt" in diag_code:
                st.markdown("---")
                st.info("📊 그래프 렌더링 확인")
                try:
                    local_vars = {}
                    # exec는 로컬 툴에서만 허용
                    exec(diag_code, globals(), local_vars)
                    if 'fig' in local_vars:
                        st.pyplot(local_vars['fig'])
                    else:
                        st.warning("코드 실행됨 (fig 객체 없음)")
                except Exception as e:
                    st.error(f"그래프 오류: {e}")

        st.markdown("---")
        
        # ==========================================
        # 3. 버튼 분리 (핵심 변경 사항)
        # ==========================================
        col_btn_save, col_btn_move = st.columns([1, 1])

        # [버튼 1] 데이터 저장만 수행 (이동 X, 리프레시 X)
        with col_btn_save:
            if st.button("💾 데이터 저장 (DB Save)", type="secondary", width="stretch"):
                if 'cropped_img' not in st.session_state:
                    st.error("이미지 세션 만료")
                else:
                    try:
                        with st.spinner("데이터베이스에 저장 중..."):
                            # 1. 이미지 업로드
                            timestamp = int(time.time())
                            clean_name = re.sub(r'[^a-zA-Z0-9가-힣_-]', '', current_file['name'].rsplit('.', 1)[0])
                            img_filename = f"{clean_name}_{timestamp}.jpg"
                            img_url = upload_image_to_storage(st.session_state['cropped_img'], img_filename)
                            
                            # 2. Firestore 저장
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
                                "content": {
                                    "problem": prob_text, 
                                    "diagram_desc": diag_desc,
                                    "diagram_code": diag_code
                                },
                                "created_at": firestore.SERVER_TIMESTAMP,
                                "labeler_version": "v3.2-split-actions"
                            }
                            db.collection("math_dataset").add(doc_data)
                            
                            # 성공 표시 (저장되었다는 플래그 설정)
                            st.session_state['is_saved'] = True
                            st.toast("✅ 데이터가 안전하게 저장되었습니다!", icon="💾")
                            st.success("저장 완료. 더 수정하거나 '완료 처리'를 눌러 넘어가세요.")
                            
                    except Exception as e:
                        st.error(f"저장 실패: {e}")

        # [버튼 2] 파일 이동 및 다음 사진 (저장 로직 없음)
        with col_btn_move:
            # 저장을 안 했으면 경고를 띄워주기 위해 help 메시지 추가
            btn_label = "✅ 완료 및 다음 파일 (Move & Next)"
            if not st.session_state.get('is_saved', False):
                btn_label += " [⚠️미저장 상태]"
            
            if st.button(btn_label, type="primary", width="stretch"):
                # 안전장치: 저장을 안 했는데 이동하려고 하면 경고
                if not st.session_state.get('is_saved', False):
                    st.warning("⚠️ 데이터를 아직 저장하지 않았습니다! 먼저 '데이터 저장'을 눌러주세요.")
                    st.stop()
                
                if done_folder_id:
                    with st.spinner("파일 정리 중..."):
                        success = move_file_to_done(current_file['id'], folder_id, done_folder_id)
                        if success:
                            st.toast("🚀 파일 이동 완료! 다음 문제로 넘어갑니다.")
                            # 로컬 리스트 업데이트
                            st.session_state['drive_files'].pop(idx)
                            # 상태 초기화
                            st.session_state.pop('cropped_img', None)
                            st.session_state.pop('extracted', None)
                            st.session_state.pop('is_saved', None) # 저장 플래그 초기화
                            
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("파일 이동 실패 (권한을 확인하세요)")
                else:
                    st.error("완료 폴더(Done Folder ID)가 설정되지 않았습니다.")

else:
    st.info("👈 드라이브 연결 필요")











