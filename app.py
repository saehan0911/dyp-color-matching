import streamlit as st
import google.generativeai as genai

# 페이지 기본 설정
st.set_page_config(
    page_title="분체도장 초정밀 CCM AI 시스템",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 분체도장 초정밀 Color Matching (CCM) AI")
st.caption("PEP-1681/E-12 하이브리드 고은폐 베이스 전용 0.01g 단위 AI 조색 보정 시스템")

# 사이드바: Gemini API 키 입력
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("Gemini API Key를 입력하세요", type="password")
    st.markdown("[👉 Gemini API Key 무료 발급받기](https://aistudio.google.com/app/apikey)")
    st.info("API Key를 입력하시면 AI 실시간 조색 계산이 활성화됩니다.")

# 1. 타겟(STD) 입력
st.subheader("1. 목표 색상 (STD Target)")
col1, col2, col3 = st.columns(3)
with col1:
    std_l = st.number_input("STD L*", value=89.63, format="%.2f")
with col2:
    std_a = st.number_input("STD a*", value=1.32, format="%.2f")
with col3:
    std_b = st.number_input("STD b*", value=11.19, format="%.2f")

st.divider()

# 2. 현재 배합 및 측정 오차 입력
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("2. 현재 차수 배합 (g)")
    batch_num = st.number_input("현재 차수 (Batch No.)", min_value=1, value=8)
    black_5 = st.number_input("HI-BLACK 50L (5%)", value=0.49, format="%.2f", step=0.01)
    yellow = st.number_input("P-261 YELLOW", value=6.63, format="%.2f", step=0.01)
    red_5 = st.number_input("RED-107 (5%)", value=1.94, format="%.2f", step=0.01)
    green_5 = st.number_input("GREEN L8710 (5%)", value=0.00, format="%.2f", step=0.01)

with col_right:
    st.subheader("3. 측정 결과 CIE 오차 (Δ)")
    dl = st.number_input("Δ L* (명도)", value=0.28, format="%.2f", step=0.01)
    da = st.number_input("Δ a* (붉은/녹색기)", value=-0.15, format="%.2f", step=0.01)
    db = st.number_input("Δ b* (노란/푸른기)", value=0.10, format="%.2f", step=0.01)

st.divider()

# 4. AI 처방 생성 버튼
if st.button("🚀 AI 초정밀 0.01g 처방 산출하기", type="primary", use_container_width=True):
    if not api_key:
        st.error("좌측 사이드바에 Gemini API Key를 입력해 주세요.")
    else:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-pro')

            # 프롬프트 생성
            prompt = f"""
            너는 분체도장 초정밀 CCM 안료 보정 전문가야.
            아래 현장 측정 데이터를 바탕으로 비선형 안료 포화 현상, 파스텔 화이트 베이스(TiO2 285g) Wash-out 효과, 5% 희석 마스터배치 민감도를 고려하여 다음 차수({batch_num + 1}번) 배합을 0.01g 단위로 처방해줘.

            [타겟 STD]
            L*: {std_l}, a*: {std_a}, b*: {std_b}

            [현재 {batch_num}번 배합]
            - HI-BLACK 50L(5%): {black_5}g
            - P-261 YELLOW: {yellow}g
            - RED-107(5%): {red_5}g
            - GREEN L8710(5%): {green_5}g

            [현재 {batch_num}번 측정 오차]
            - dL: {dl}
            - da: {da}
            - db: {db}

            다음 양식으로 명확하게 답변해줘:
            1. 오차 분석 및 물리적 현상 원인 (파스텔 Wash-out / 안료 상호작용)
            2. {batch_num + 1}번 0.01g 초정밀 배합 처방 (Black, Yellow, Red, Green 각각 변경량과 최종g)
            3. 보정 근거 및 0.01g 산출 이유
            """

            with st.spinner("Gemini AI가 도막 물리 반응 및 0.01g 안료 밸런스를 계산 중입니다..."):
                response = model.generate_content(prompt)
                st.success("처방 산출이 완료되었습니다!")
                st.markdown(response.text)

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")

