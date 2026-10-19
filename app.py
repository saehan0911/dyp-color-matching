import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
# ---------------------------------------------------------
# 1. 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="DYP 기술연구소 | AI 초정밀 CCM 시스템",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ---------------------------------------------------------
# 2. 커스텀 CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    .stApp { background-color: #F8FAFC; }
   
    .enterprise-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #0F172A 100%);
        border-bottom: 4px solid #2563EB;
        padding: 20px 28px;
        border-radius: 16px;
        color: #FFFFFF;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
        margin-bottom: 20px;
    }
    .enterprise-title {
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .enterprise-subtitle {
        color: #94A3B8;
        font-size: 0.92rem;
        margin-top: 4px;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .badge-red { background: rgba(239, 68, 68, 0.2); color: #FCA5A5; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-blue { background: rgba(59, 130, 246, 0.2); color: #93C5FD; border: 1px solid rgba(59, 130, 246, 0.4); }
   
    .cat-header {
        padding: 8px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.98rem;
        color: #FFFFFF;
        margin-top: 4px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .cat-1-bg { background: linear-gradient(90deg, #1E40AF, #2563EB); }
    .cat-2-bg { background: linear-gradient(90deg, #D97706, #F59E0B); }
    .cat-3-bg { background: linear-gradient(90deg, #059669, #10B981); }
    .cat-4-bg { background: linear-gradient(90deg, #6D28D9, #8B5CF6); }
   
    .ai-info-box {
        background-color: #F1F5F9;
        border-left: 3px solid #10B981;
        padding: 6px 10px;
        border-radius: 0 4px 4px 0;
        font-size: 0.82rem;
        color: #334155;
        margin-top: 2px;
        margin-bottom: 8px;
    }
    .kpi-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        margin: 16px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .kpi-label { font-size: 0.88rem; color: #64748B; font-weight: 600; }
    .kpi-val { font-size: 1.8rem; font-weight: 800; color: #1E293B; }
    .stButton > button { border-radius: 6px !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)
# 메인 헤더
st.markdown("""
<div class="enterprise-header">
    <div class="enterprise-title">
        🧪 DYP주식회사 기술연구소 조색프로그램
    </div>
    <div class="enterprise-subtitle">
        환경친화형 분체도료 AI 초정밀(0.01g) Color Matching & Real-time Reasoning System
    </div>
    <div style="display:flex; gap:10px; margin-top:10px; flex-wrap:wrap;">
        <span class="badge-pill badge-red">🔒 영업비밀 엄수 (Confidential)</span>
        <span class="badge-pill badge-blue">👤 제작자: 기술연구소 최새한 대리</span>
    </div>
</div>
""", unsafe_allow_html=True)
# ---------------------------------------------------------
# 3. 모델 및 프롬프트 설정 (Gemini 1.5 Pro)
# ---------------------------------------------------------
MODEL_NAME = "gemini-1.5-pro"
SYSTEM_PROMPT = """
너는 DYP주식회사 기술연구소의 수석 연구원급 AI 분석가(Gemini 1.5 Pro 엔진)야.
[현존 분체도료 공식 및 이론 탑재]
1. Kubelka-Munk (K/S) 이론: 도막의 반사율과 안료 농도의 비선형적 관계 (산란/흡광 포화 모델).
2. CIE L*a*b* 색차 공식 (ΔE*ab, ΔE_CMC).
3. Saunderson 표면 반사 보정 (내부/외부 반사율).
4. 희석 안료 고착색성: 5% 마스터배치 등 미세 안료의 0.01g 단위 민감도 곡선.
5. 파스텔 베이스 Wash-out 효과 및 소성 오븐 황변(+b*) 고려.
[성장 프로그램 및 인간 실수 감안]
- 너는 입력되는 누적 피드백 데이터를 바탕으로 패턴을 분석하여 지속적으로 진화하고 똑똑해져야 한다.
- 단, 데이터가 항상 100% 정답은 아니다. 인간은 계량 실수, 조색 과정의 오차, 오븐 온도 편차 등을 발생시킬 수 있음을 명심해라.
- 따라서, 하나의 튀는 결과치(이상치)에 맹신하여 흔들리지 말고, 전체적인 트렌드와 합리성을 필터링하여 성장해라.
[작동 원칙]
- 계산을 요구받을 경우: 인사말 없이 "1. 🎯 보정 배합 처방" 마크다운 표를 최상단에 출력하라.
- 질문을 받을 경우: 뜬금없이 처방이나 계량을 하지 말고, 오직 질문에 대해서만 이론적/실무적 답변을 하라.
"""
# ---------------------------------------------------------
# 4. 파일 입출력 및 세션 초기화
# ---------------------------------------------------------
FEEDBACK_FILE = "ai_feedback_data.csv"
def get_feedback_df():
    if os.path.exists(FEEDBACK_FILE):
        try:
            return pd.read_csv(FEEDBACK_FILE)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()
def save_feedback_df(df):
    df.to_csv(FEEDBACK_FILE, index=False, encoding="utf-8-sig")
if "materials" not in st.session_state:
    st.session_state.materials = {
        "cat1": [{"name": "PEP-1681", "desc": "폴리에스터 수지 (주 바인더 수지계)", "amount": 360.00}],
        "cat2": [{"name": "LA-100", "desc": "유동성 향상 및 대전 특성제어 첨가제", "amount": 10.00}],
        "cat3": [{"name": "TiO2 R-996", "desc": "이산화티타늄 고은폐 백색 안료", "amount": 285.00}],
        "cat4": [{"name": "P-261 YELLOW", "desc": "황색 유기 안료", "amount": 6.67}]
    }
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
def get_validated_api_key(manual_key_input=""):
    key_candidate = manual_key_input.strip() if manual_key_input.strip() else str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    if not key_candidate:
        return None, "API Key가 설정되지 않았습니다."
    if len(key_candidate) < 10:
        return None, "API Key가 너무 짧습니다."
    return key_candidate, "OK"
# ---------------------------------------------------------
# 5. 사이드바
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ 시스템 설정")
    user_key_input = st.text_input("🔑 API Key 입력", type="password")
    active_api_key, key_status_msg = get_validated_api_key(user_key_input)
   
    if active_api_key:
        st.success(f"🟢 Gemini Pro API 검증 완료")
    else:
        st.error(f"🔴 API Key 오류")
# ---------------------------------------------------------
# 6. 메인 UI: 목표 색상 및 전체 배합 (실시간 갱신 적용)
# ---------------------------------------------------------
with st.container(border=True):
    st.markdown("#### 🎯 1. 목표 색상 (STD Target)")
    c1, c2, c3 = st.columns(3)
    std_l = c1.number_input("STD L*", value=89.63, format="%.2f")
    std_a = c2.number_input("STD a*", value=1.32, format="%.2f")
    std_b = c3.number_input("STD b*", value=11.19, format="%.2f")
current_formula = {}
def render_category(cat_key, cat_title, cat_css_class):
    st.markdown(f'<div class="cat-header {cat_css_class}"><span>{cat_title}</span></div>', unsafe_allow_html=True)
   
    items_to_remove = []
    for idx, item in enumerate(st.session_state.materials[cat_key]):
        c_name, c_amt, c_del = st.columns([2.2, 1.8, 1])
       
        # 실시간 이름 변경 감지 및 AI 검색결과 갱신
        new_name = c_name.text_input(f"재료명 #{idx+1}", value=item["name"], key=f"name_{cat_key}_{idx}")
        new_amt = c_amt.number_input(f"투입량(g) #{idx+1}", value=float(item["amount"]), format="%.2f", step=0.01, key=f"amt_{cat_key}_{idx}")
       
        item["amount"] = new_amt
       
        if new_name != item["name"]:
            item["name"] = new_name
            item["desc"] = "검색 중..."
            if active_api_key and new_name.strip():
                try:
                    genai.configure(api_key=active_api_key)
                    m = genai.GenerativeModel(MODEL_NAME)
                    item["desc"] = m.generate_content(f"분체도료 원료 '{new_name}'의 역할을 1줄로 요약").text.strip()
                except:
                    item["desc"] = "AI 검색 실패"
            elif not new_name.strip():
                item["desc"] = ""
            st.rerun()
        if c_del.button("🗑️", key=f"del_{cat_key}_{idx}"):
            items_to_remove.append(idx)
       
        st.markdown(f"""
        <div class="ai-info-box">
            💡 <b>AI 검색결과:</b> {item.get('desc', '')}
        </div>
        """, unsafe_allow_html=True)
       
        if item["name"].strip():
            current_formula[item["name"]] = item["amount"]
           
    if items_to_remove:
        for i in sorted(items_to_remove, reverse=True):
            st.session_state.materials[cat_key].pop(i)
        st.rerun()
    if st.button(f"➕ 재료 추가", key=f"add_btn_{cat_key}"):
        st.session_state.materials[cat_key].append({"name": "", "desc": "재료명을 입력하면 AI가 설명합니다.", "amount": 0.00})
        st.rerun()
with st.container(border=True):
    st.markdown("#### 🧪 2. 전체 배합")
    c_r1_1, c_r1_2 = st.columns(2)
    with c_r1_1: render_category("cat1", "🏢 수지 & 경화제", "cat-1-bg")
    with c_r1_2: render_category("cat2", "⚡ 기능성 첨가제", "cat-2-bg")
    st.divider()
    c_r2_1, c_r2_2 = st.columns(2)
    with c_r2_1: render_category("cat3", "⚪ 은폐 & 체질 안료", "cat-3-bg")
    with c_r2_2: render_category("cat4", "🎨 유색 착색 안료", "cat-4-bg")
total_weight = sum(current_formula.values())
st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-label">전체 설정 처방 총중량</div>
    <div class="kpi-val" style="color:#2563EB;">{total_weight:.2f} <span style="font-size:1.1rem; color:#64748B;">g</span></div>
</div>
""", unsafe_allow_html=True)
# ---------------------------------------------------------
# 7. 오차 입력 및 보정 산출 (독립)
# ---------------------------------------------------------
with st.container(border=True):
    st.markdown("#### 📐 3. 측정 결과 (CIE 오차 Δ)")
    c_b, c_l, c_a, c_b_axis = st.columns(4)
    batch_num = c_b.number_input("배치 차수", min_value=1, value=1)
    dl = c_l.number_input("Δ L*", value=0.08, format="%.2f", step=0.01)
    da = c_a.number_input("Δ a*", value=-0.03, format="%.2f", step=0.01)
    db = c_b_axis.number_input("Δ b*", value=0.02, format="%.2f", step=0.01)
    curr_de = round(float(np.sqrt(dl**2 + da**2 + db**2)), 3)
    st.caption(f"**현재 색차 (ΔE):** {curr_de:.3f}")
if st.button("🚀 동적 처방 보정 연산 실행", type="primary", use_container_width=True):
    if not active_api_key:
        st.error("API Key를 확인해주세요.")
    else:
        try:
            fb_df = get_feedback_df()
            fb_text = "누적 피드백 없음" if fb_df.empty else "\n".join([f"- {r['mat']} {r['dg']}g 조절 -> ΔL:{r['dl']}, Δa:{r['da']}, Δb:{r['db']}" for _, r in fb_df.tail(20).iterrows()])
           
            genai.configure(api_key=active_api_key)
            model = genai.GenerativeModel(MODEL_NAME)
            f_str = "\n".join([f"- {k}: {v:.2f}g" for k, v in current_formula.items()])
            prompt = f"""
            {SYSTEM_PROMPT}
            [누적 피드백 학습 데이터 (인간 실수 필터링 감안)]
            {fb_text}
            [현재 상황]
            목표: L* {std_l}, a* {std_a}, b* {std_b}
            현재 배합: {f_str}
            오차: ΔL* {dl}, Δa* {da}, Δb* {db}
            [지시] 1번 마크다운 표를 최상단에 즉시 출력하라.
            """
            with st.spinner("🧠 Gemini 1.5 Pro가 공식 및 누적 데이터를 연산 중입니다..."):
                res = model.generate_content(prompt)
                st.session_state.last_calc = res.text
                st.success("연산 완료!")
        except Exception as e:
            st.error(f"오류: {e}")
if "last_calc" in st.session_state:
    st.markdown("---")
    st.markdown("### 📋 AI 초정밀 보정 결과")
    st.markdown(st.session_state.last_calc)
# ---------------------------------------------------------
# 8. AI 질문하기 (완전 독립)
# ---------------------------------------------------------
st.divider()
with st.container(border=True):
    st.markdown("#### 💬 AI 질문하기 (기술 상담 전용)")
    st.caption("질문만 처리하며, 뜬금없는 처방/계량을 진행하지 않습니다.")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if q := st.chat_input("질문을 입력하세요"):
        if active_api_key:
            st.chat_message("user").markdown(q)
            st.session_state.chat_history.append({"role": "user", "content": q})
            genai.configure(api_key=active_api_key)
            model = genai.GenerativeModel(MODEL_NAME)
            qa_prompt = f"{SYSTEM_PROMPT}\n[사용자 질문]\n{q}\n[주의] 배합 처방표를 절대 만들지 말고 기술적 답변만 하라."
            with st.chat_message("assistant"):
                with st.spinner("답변 중..."):
                    a = model.generate_content(qa_prompt).text
                    st.markdown(a)
                    st.session_state.chat_history.append({"role": "assistant", "content": a})
# ---------------------------------------------------------
# 9. 실측 데이터 누적 학습 (피드백 추가)
# ---------------------------------------------------------
st.divider()
with st.container(border=True):
    st.markdown("#### 📈 데이터 누적 학습 (실측 피드백 추가)")
    st.caption("실제 투입 결과 데이터를 입력하면 AI가 실수를 필터링하며 능력을 강화합니다.")
   
    col1, col2, col3, col4, col5 = st.columns(5)
    f_mat = col1.selectbox("원료", options=list(current_formula.keys()) if current_formula else ["없음"])
    f_dg = col2.number_input("변화량(g)", value=0.00, step=0.01)
    f_dl = col3.number_input("결과 ΔL*", value=0.00, step=0.01)
    f_da = col4.number_input("결과 Δa*", value=0.00, step=0.01)
    f_db = col5.number_input("결과 Δb*", value=0.00, step=0.01)
   
    if st.button("데이터 저장 및 학습 반영", use_container_width=True):
        new_data = {"mat": f_mat, "dg": f_dg, "dl": f_dl, "da": f_da, "db": f_db}
        df = get_feedback_df()
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        save_feedback_df(df)
        st.success("데이터가 누적되어 AI 성장 프로그램에 반영되었습니다.")
       
    df_show = get_feedback_df()
    if not df_show.empty:
        with st.expander("누적 학습 데이터 보기"):
            st.dataframe(df_show, use_container_width=True)
