import importlib
import streamlit as st

from a_streamlit_env import a_chart_realtime
from a_streamlit_env import a_AI_chat
from a_streamlit_env import a_kakao_결제
from a_streamlit_env import a_homepage
from a_streamlit_env import a_market

# 에이전틱 트레이딩
# 과거 기간 이슈 모의투자 시뮬레이션 (월 2000원)
# 이슈기반으로 지표 만들어주고, 이걸로 백테스트 기능 제공 (월 2000 ?)
# 이슈기반 커스텀 룰베 전략 운용(실투자) = 전략당 책정 , 월 x 원 (이건 나중)
# AI 속보알람 - 관심종목과 관련된 이슈만 AI가 정리하고 분석해서 카톡으로 알려줌, 속보 포함(결제) = 월 3900 베이직
# AI 전략생성(llm) = 만드는건 3회까지 무료, 실전 사용료 책정
# AI 매매일지 작성 , 사용자 행동 분석
#######################################################################
# 이슈 정보 웹에서 알려주기 + Q&A 웹채팅 = 공짜 (시간마다 업로드, 단체 채팅)
# 실시간 모의투자 시뮬레이션 기능 = 공짜

st.set_page_config(layout="wide")


# 사이드바 메뉴 설정
st.sidebar.title("메뉴")
page = st.sidebar.radio("페이지 선택", ["홈페이지","이슈 시뮬레이터/AI 채팅", "나만의 AI 투자 전략", "상점"])


if page == '홈페이지':
    a_homepage.start()


elif page == "이슈 시뮬레이터/AI 채팅":

    st.header("실시간 차트")
    a_chart_realtime.start(a_AI_chat)
    st.session_state.initialized["is_not_init"] = True #초기 실행인지 아닌지 표시(초기아니면 True)

elif page == "실시간 AI 애널리스트":
    'CRAG LLM 기술 사용한거 여기 넣어야함'
    '인터넷정보+재무정보+뉴스 사용해서 맞춤형 실시간 보고서 생성'
    '사용자가 어떤 질문을해도 보고서 형태로 알려줌'


elif page == "나만의 AI 투자 전략":
    '사용자가 말만하면 AI가 전략 생성 해줌'
    '이슈데이터 활용한 전략 생성가능'


elif page == "상점":
    a_market.start()

