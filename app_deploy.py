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
from streamlit_cropper import st_cropper

# ==========================================
# 0. 사용자 설정
# ==========================================
BUCKET_NAME = "math-problem-collector.firebasestorage.app"

# ==========================================
# 1. Configuration & Auth
# ==========================================
st.set_page_config(layout="wide", page_title="Cloud Math Factory")

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
    return Image.open(file_obj)

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

# [핵심] AI 자동 분할 (Bounding Box)
def auto_split_gemini(image, count):
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        return {"error": "API Key Missing"}

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp") # 2.0이 좌표를 더 잘 잡음 (안되면 1.5-pro 사용)
        
        prompt = f"""
        Detect exactly {count} math problems in this image.
        Return a JSON list of bounding boxes in [ymin, xmin, ymax, xmax] format (scale 0-1000).
        Also extract the LaTeX content for each.
        
        Output JSON format:
        [
            {{
                "box_2d": [ymin, xmin, ymax, xmax],
                "problem_text": "LaTeX...",
                "diagram_desc": "Description..."
            }}
        ]
        """
        response = model.generate_content([prompt, image])
        text = response.text.replace("```json", "").replace("```", "")
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}

# 단일 분석용
def extract_single(image):
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(["Convert to LaTeX JSON: {'problem_text': '...', 'diagram_desc': '...'}", image])
    return json.loads(response.text.replace("```json", "").replace("```", ""))

# ==========================================
# 3. Main UI
# ==========================================
st.title("🏭 Cloud Math Factory (AI Auto-Split)")

with st.sidebar:
    st.header("⚙️ 작업 설정")
    default_folder = st.secrets.get("DEFAULT_FOLDER_ID", "")
    done_folder_default = st.secrets.get("DONE_FOLDER_ID", "")
    
    folder_id = st.text_input("작업 폴더 ID", value=default_folder)
    done_folder_id = st.text_input("완료 폴더 ID", value=done_folder_default)
    
    if st.button("📂 드라이브 스캔"):
        if folder_id:
            files = list_drive_images(folder_id)
            st.session_state['drive_files'] = files
            st.session_state['idx'] = 0
            st.success(f"{len(files)}개 발견")

    st.markdown("---")
    # 네비게이션
    c_prev, c_next = st.columns(2)
    if c_prev.button("◀ 이전"):
        if st.session_state.get('idx', 0) > 0:
            st.session_state['idx'] -= 1
            st.session_state.pop('split_results', None)
            st.rerun()
    if c_next.button("다음 ▶"):
        if st.session_state.get('drive_files') and st.session_state['idx'] < len(st.session_state['drive_files']) - 1:
            st.session_state['idx'] += 1
            st.session_state.pop('split_results', None)
            st.rerun()

if 'drive_files' in st.session_state and st.session_state['drive_files']:
    files = st.session_state['drive_files']
    idx = st.session_state['idx']
    
    if idx >= len(files):
        st.info("모든 파일 처리 완료!")
        st.stop()
        
    current_file = files[idx]
    
    # 이미지 로드
    if 'current_file_id' not in st.session_state or st.session_state['current_file_id'] != current_file['id']:
        st.session_state['original_img'] = download_image_from_drive(current_file['id'])
        st.session_state['current_file_id'] = current_file['id']
        st.session_state.pop('split_results', None) # 새 파일이면 결과 초기화

    original_img = st.session_state['original_img']
    
    # ==========================================
    # [모드 선택] 수동 vs 자동
    # ==========================================
    mode = st.radio("작업 모드", ["✂️ 수동 자르기 (Manual)", "🤖 AI 자동 분할 (Auto-Split)"], horizontal=True)
    
    if mode == "✂️ 수동 자르기 (Manual)":
        col_L, col_R = st.columns([1.5, 1])
        with col_L:
            cropped_img = st_cropper(original_img, realtime_update=True, box_color='red', aspect_ratio=None)
        with col_R:
            if st.button("이 영역 분석 및 저장"):
                with st.spinner("분석 중..."):
                    extracted = extract_single(cropped_img)
                    st.session_state['manual_data'] = extracted
                    st.session_state['manual_crop'] = cropped_img
            
            if 'manual_data' in st.session_state:
                data = st.session_state['manual_data']
                with st.form("manual_save"):
                    # (입력 필드들 - 간소화)
                    c1, c2 = st.columns(2)
                    subj = c1.selectbox("과목", OPTIONS['subject'])
                    grd = c2.selectbox("학년", OPTIONS['grade'])
                    src = c1.selectbox("출처", OPTIONS['source_org'])
                    unt = c2.selectbox("단원", OPTIONS['unit_major'])
                    dif = c1.selectbox("난이도", OPTIONS['difficulty'])
                    typ = c2.selectbox("유형", OPTIONS['question_type'])
                    cpt = st.selectbox("개념", OPTIONS['concepts'])
                    
                    prob = st.text_area("문제", data.get('problem_text', ""))
                    desc = st.text_area("설명", data.get('diagram_desc', ""))
                    
                    if st.form_submit_button("🔥 저장 (이동 X)"):
                        # (저장 로직 - 위에 있는 코드와 동일하게 구현)
                        timestamp = int(time.time())
                        clean_name = current_file['name'].rsplit('.', 1)[0]
                        img_filename = f"{clean_name}_{timestamp}.jpg"
                        img_url = upload_image_to_storage(st.session_state['manual_crop'], img_filename)
                        
                        doc_data = {
                            "original_filename": current_file['name'],
                            "drive_file_id": current_file['id'],
                            "image_url": img_url,
                            "storage_path": f"cropped_problems/{img_filename}",
                            "meta": {"subject": subj, "grade": grd, "source": src, "unit": unt, "difficulty": dif, "question_type": typ, "concept": cpt},
                            "content": {"problem": prob, "diagram": desc},
                            "created_at": firestore.SERVER_TIMESTAMP
                        }
                        db.collection("math_dataset").add(doc_data)
                        st.success("저장 완료!")

    else: # 🤖 Auto-Split Mode
        st.info("이 이미지에 문제가 몇 개 있나요?")
        prob_count = st.number_input("문제 개수", min_value=1, max_value=10, value=2)
        
        if st.button("🚀 AI 자동 분할 실행"):
            with st.spinner("AI가 문제 영역을 찾고 있습니다..."):
                results = auto_split_gemini(original_img, prob_count)
                
                if isinstance(results, list):
                    # 좌표로 이미지 자르기
                    width, height = original_img.size
                    processed_items = []
                    
                    for item in results:
                        box = item['box_2d'] # [ymin, xmin, ymax, xmax] 0-1000
                        ymin, xmin, ymax, xmax = box
                        # 좌표 변환
                        left = xmin / 1000 * width
                        top = ymin / 1000 * height
                        right = xmax / 1000 * width
                        bottom = ymax / 1000 * height
                        
                        crop = original_img.crop((left, top, right, bottom))
                        processed_items.append({"img": crop, "data": item})
                    
                    st.session_state['split_results'] = processed_items
                else:
                    st.error(f"실패: {results}")

        # 결과 탭 표시
        if 'split_results' in st.session_state:
            items = st.session_state['split_results']
            tabs = st.tabs([f"문제 {i+1}" for i in range(len(items))])
            
            # 전체 저장용 데이터
            save_queue = []
            
            for i, tab in enumerate(tabs):
                with tab:
                    item = items[i]
                    col_img, col_form = st.columns([1, 1.5])
                    with col_img:
                        st.image(item['img'], caption=f"Auto-Crop {i+1}")
                    with col_form:
                        # 폼 입력 (여기서 수정 가능)
                        c1, c2 = st.columns(2)
                        subj = c1.selectbox("과목", OPTIONS['subject'], key=f"s_{i}")
                        grd = c2.selectbox("학년", OPTIONS['grade'], key=f"g_{i}")
                        src = c1.selectbox("출처", OPTIONS['source_org'], key=f"src_{i}")
                        unt = c2.selectbox("단원", OPTIONS['unit_major'], key=f"u_{i}")
                        dif = c1.selectbox("난이도", OPTIONS['difficulty'], key=f"d_{i}")
                        typ = c2.selectbox("유형", OPTIONS['question_type'], key=f"t_{i}")
                        cpt = st.selectbox("개념", OPTIONS['concepts'], key=f"c_{i}")
                        
                        prob = st.text_area("문제", item['data'].get('problem_text', ""), key=f"p_{i}", height=100)
                        desc = st.text_area("설명", item['data'].get('diagram_desc', ""), key=f"dsc_{i}", height=50)
                        
                        # 이 탭의 데이터 저장용 딕셔너리 생성
                        save_queue.append({
                            "img": item['img'],
                            "meta": {"subject": subj, "grade": grd, "source": src, "unit": unt, "difficulty": dif, "question_type": typ, "concept": cpt},
                            "content": {"problem": prob, "diagram": desc}
                        })
            
            st.divider()
            if st.button("💾 전체 저장 및 완료처리 (Save All & Move)", type="primary"):
                with st.spinner("일괄 저장 중..."):
                    for idx, data in enumerate(save_queue):
                        # 이미지 업로드
                        timestamp = int(time.time())
                        clean_name = current_file['name'].rsplit('.', 1)[0]
                        img_filename = f"{clean_name}_{timestamp}_{idx}.jpg"
                        img_url = upload_image_to_storage(data['img'], img_filename)
                        
                        # DB 저장
                        doc_data = {
                            "original_filename": current_file['name'],
                            "drive_file_id": current_file['id'],
                            "problem_index": idx + 1,
                            "image_url": img_url,
                            "storage_path": f"cropped_problems/{img_filename}",
                            "meta": data['meta'],
                            "content": data['content'],
                            "created_at": firestore.SERVER_TIMESTAMP
                        }
                        db.collection("math_dataset").add(doc_data)
                    
                    # 파일 이동
                    if done_folder_id:
                        move_file_to_done(current_file['id'], folder_id, done_folder_id)
                        st.toast("완벽합니다! 저장하고 파일을 치웠습니다.")
                        time.sleep(1)
                        st.session_state.pop('split_results', None)
                        st.rerun()
                    else:
                        st.success("저장 완료! (폴더 이동은 안 함)")

else:
    st.info("왼쪽 사이드바에서 드라이브를 연결해주세요.")
