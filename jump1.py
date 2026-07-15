import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import plotly.express as px
import cv2
import mediapipe as mp
import numpy as np
import os
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# ==========================================
# 1. 환경 설정 및 API 키
# ==========================================
GOOGLE_SHEET_KEY = "1s1XcEb-7gU4r024eQJXsvDqIab6MeSP9GjpBbP4QShE"

# 스마트폰 통신을 위한 RTC 서버(STUN) 설정 (구글의 무료 서버 사용)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ==========================================
# 2. 구글 시트 연동 함수
# ==========================================
@st.cache_resource
def get_gspread_client():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "secrets.json")
        creds = ServiceAccountCredentials.from_json_keyfile_name(json_path, scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"구글 시트 인증 실패: {e}")
        return None


def get_all_sheet_names():
    client = get_gspread_client()
    if client:
        try:
            doc = client.open_by_key(GOOGLE_SHEET_KEY)
            return [worksheet.title for worksheet in doc.worksheets()]
        except Exception as e:
            st.error(f"시트 목록 불러오기 실패: {e}")
    return ["녹산초3-7반 데이터"]


def load_class_data(sheet_name):
    client = get_gspread_client()
    if client:
        try:
            doc = client.open_by_key(GOOGLE_SHEET_KEY)
            sheet = doc.worksheet(sheet_name)
            data = sheet.get_all_records()
            return pd.DataFrame(data)
        except Exception as e:
            st.error(f"데이터 불러오기 실패: {e}")
    return pd.DataFrame()


def update_jump_rope_count(sheet_name, student_name, count):
    client = get_gspread_client()
    if client:
        try:
            doc = client.open_by_key(GOOGLE_SHEET_KEY)
            sheet = doc.worksheet(sheet_name)
            cell = sheet.find(student_name)
            row_idx = cell.row
            sheet.update_cell(row_idx, 12, count)  # L열 (12번째) 업데이트
            return True
        except Exception as e:
            st.error(f"구글 시트 저장 실패: {e}")
    return False


# ==========================================
# 3. WebRTC 카메라 영상 처리 클래스
# ==========================================
# 클라우드 서버가 아닌 '접속한 사람의 기기'에서 프레임을 쪼개서 AI 분석을 돌립니다.
class PoseProcessor(VideoTransformerBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.jump_state = "down"
        self.counter = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # 전면 카메라 사용 시 거울 모드를 위해 좌우 반전
        img = cv2.flip(img, 1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 관절 위치 분석
        results = self.pose.process(rgb_img)

        # 관절들이 인식되었을 때 뼈대 그리기 및 수직 높이 감지
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(rgb_img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark
            left_hip_y = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y
            right_hip_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y
            avg_hip_y = (left_hip_y + right_hip_y) / 2

            threshold_up = 0.52
            threshold_down = 0.56

            if avg_hip_y < threshold_up and self.jump_state == "down":
                self.jump_state = "up"
            elif avg_hip_y > threshold_down and self.jump_state == "up":
                self.counter += 1
                self.jump_state = "down"

        # 영상 화면 상단에 AI 실시간 카운트 값 합성 (BGR 형태로 다시 변환 후 송출)
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr_img, f"AI Count: {self.counter}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 0), 4, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(bgr_img, format="bgr24")


# ==========================================
# 4. Streamlit UI 구성
# ==========================================
st.set_page_config(page_title="AI 줄넘기 배틀", page_icon="⚡", layout="wide")

st.title("⚡ 실시간 AI 줄넘기 배틀 대시보드 (모바일 호환)")
st.write("스마트폰 카메라를 세워두고 줄넘기를 뛰면 AI가 개수를 자동으로 세어줍니다!")

# 좌측 사이드바: 학반 선택
st.sidebar.header("🏫 학반 선택")
sheet_list = get_all_sheet_names()
selected_class = st.sidebar.selectbox("우리 반을 선택하세요", sheet_list)

# 데이터 로드 및 전처리
df = load_class_data(selected_class)

if not df.empty:
    df['줄넘기 횟수'] = pd.to_numeric(df['줄넘기 횟수'], errors='coerce').fillna(0).astype(int)
    df['권장 줄넘기 횟수'] = pd.to_numeric(df['권장 줄넘기 횟수'], errors='coerce').fillna(0).astype(int)
    df['모둠'] = df['모둠'].astype(str).str.strip().replace('', '미정')

    col_cam, col_dash = st.columns([1.1, 1])

    # ----------------------------------------------------
    # [좌측 열] WebRTC AI 카메라 영역
    # ----------------------------------------------------
    with col_cam:
        st.markdown("### 📸 AI 실시간 모바일 카운터")

        student_list = df['데이터용 성명(함수 복사해서 사용)'].tolist()
        selected_student = st.selectbox("기록을 등록할 학생 이름 선택", student_list)

        current_count = df[df['데이터용 성명(함수 복사해서 사용)'] == selected_student]['줄넘기 횟수'].values[0]
        st.caption(f"현재 구글 시트에 등록된 기록: **{current_count}** 회")

        st.info("아래의 **'START'** 버튼을 누르면 브라우저가 카메라 권한을 요청합니다.")

        # WebRTC 카메라 송출 인터페이스 (스마트폰 호환)
        ctx = webrtc_streamer(
            key="jump-rope",
            video_processor_factory=PoseProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )

        # 카메라가 켜져 있고, 분석이 진행 중일 때 기록 전송 UI 활성화
        if ctx.video_processor:
            current_ai_count = ctx.video_processor.counter
            st.markdown(f"#### 🎯 현재 AI 측정 기록: **{current_ai_count}** 개")

            if st.button("🔄 카운트 리셋"):
                ctx.video_processor.counter = 0
                st.rerun()

            if st.button("🔥 기록 최종 제출 및 시트 동기화", type="primary", use_container_width=True):
                with st.spinner("구글 시트에 실시간 기록 중..."):
                    if update_jump_rope_count(selected_class, selected_student, current_ai_count):
                        st.success(f"성공! {selected_student} 학생의 기록이 {current_ai_count}회로 최종 반영되었습니다!")
                        st.balloons()
                        ctx.video_processor.counter = 0  # 전송 완료 후 카운터 초기화
                        st.rerun()

    # ----------------------------------------------------
    # [우측 열] 대시보드 시각화 영역
    # ----------------------------------------------------
    with col_dash:
        st.markdown("### 🏆 실시간 모둠 및 개인 랭킹")

        total_class_jumps = df['줄넘기 횟수'].sum()
        avg_class_jumps = int(df['줄넘기 횟수'].mean())

        m1, m2 = st.columns(2)
        m1.metric("🔥 우리 반 누적 개수", f"{total_class_jumps} 회")
        m2.metric("🏃 우리 반 평균 개수", f"{avg_class_jumps} 회")

        tab1, tab2 = st.tabs(["👥 실시간 모둠 대항전", "👑 개인전 TOP 5"])

        with tab1:
            group_df = df[df['모둠'] != '미정'].groupby('모둠')['줄넘기 횟수'].mean().reset_index()
            group_df['줄넘기 횟수'] = group_df['줄넘기 횟수'].round(1)
            group_df = group_df.sort_values(by='줄넘기 횟수', ascending=False)

            if not group_df.empty:
                fig = px.bar(
                    group_df,
                    x='줄넘기 횟수',
                    y='모둠',
                    orientation='h',
                    text='줄넘기 횟수',
                    color='모둠',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_layout(xaxis_title="평균 줄넘기 횟수", yaxis_title="모둠", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("시트에 모둠(J열)을 기입해 주시면 실시간 모둠 랭킹 그래프가 연동됩니다.")

        with tab2:
            rank_df = df[['데이터용 성명(함수 복사해서 사용)', '줄넘기 횟수']].sort_values(by='줄넘기 횟수', ascending=False).head(5)
            rank_df.columns = ['이름', '횟수 (회)']
            rank_df.index = range(1, len(rank_df) + 1)

            medals = ["🥇 1등", "🥈 2등", "🥉 3등", "🏅 4등", "🏅 5등"]
            rank_df.insert(0, '순위', medals[:len(rank_df)])
            st.table(rank_df.set_index('순위'))

else:
    st.warning("데이터를 불러오지 못했습니다. 스프레드시트의 ID와 시트 구성을 다시 한번 확인해 주세요.")