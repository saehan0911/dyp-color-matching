import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import os

# 1. 페이지 기본 설정 (와이드 레이아웃 & 커스텀 타이틀)
st.set_page_config(
    page_title="DYP주식회사 기술연구소 | AI 초정밀 CCM 조색 시스템",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 커스텀 CSS (대시보드 테마)
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* 메인 엔터프라이즈 헤더 */
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
    
    /* 상단 상태 뱃지 */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .badge-red {
        background: rgba(239, 68, 68, 0.2);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-blue {
        background: rgba(59, 130, 246, 0.2);
        color: #93C5FD;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.2);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    /* 카테고리 섹션 헤더 */
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
    
    /* AI 역할 카드 */
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
    
    /* 총중량 대시보드 KPI 카드 */
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
    .kpi-label {
        font-size: 0.88rem;
        color: #64748B;
        font-weight: 600;
    }
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1E293B;
    }

    .stButton > button {
        border-radius: 6px !important;
        font-weight: 700 !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# 메인 상단 헤더
st.markdown("""
<div class="enterprise-header">
    <div class="enterprise-title">
        🧪 DYP주식회사 기술연구소 조색프로그램
    </div>
    <div class="enterprise-subtitle">
        환경친화형 분체도료 AI 초정밀(0.01g) Color Matching & Real-time Reasoning Q&A System
    </div>
    <div style="display:flex; gap:10px; margin-top:10px; flex-wrap:wrap;">
        <span class="badge-pill badge-red">🔒 영업비밀 엄수 (Confidential)</span>
        <span class="badge-pill badge-blue">👤 제작자: 기술연구소 최새한 대리</span>
        <span class="badge-pill badge-green">🟢 AI Pro Engine Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

LIBRARY_FILE = "material_library.json"
HISTORY_FILE = "color_history_dynamic.csv"

# 마스터 라이브러리 기본값
DEFAULT_LIBRARY = {
    "cat1": [
        {"name": "PEP-1681", "desc": "폴리에스터 수지 (주 바인더 수지계)"},
        {"name": "E-12", "desc": "에폭시 수지 (경화 및 도막 내화학성 부여)"},
        {"name": "HM3328", "desc": "경화제 및 레벨링 조절 보조제"},
        {"name": "PE-500", "desc": "고유연성 폴리에스터 수지"},
        {"name": "EP-101", "desc": "고밀도 에폭시 수지"}
    ],
    "cat2": [
        {"name": "LA-100", "desc": "유동성 향상 및 대전 특성제어 첨가제"},
        {"name": "BENZOIN", "desc": "도막 소성 시 가스 기포 방출 탈포제"},
        {"name": "PA-103", "desc": "표면 오렌지필 방지 레벨링 조정제"},
        {"name": "ACRONAL 4F", "desc": "표면 평활성 향상 흐름제"},
        {"name": "WAX-50", "desc": "내스크래치성 슬립 첨가제"}
    ],
    "cat3": [
        {"name": "TiO2 R-996", "desc": "이산화티타늄 고은폐 백색 안료 (빛 반사율 제어)"},
        {"name": "CaCO3", "desc": "탄산칼슘 체질재 (도막 스페이스 및 소광/비용 조절)"},
        {"name": "BaSO4", "desc": "황산바륨 고광택 체질재"},
        {"name": "Talc-325", "desc": "탈크 체질재 (소광 및 흐름방지)"}
    ],
    "cat4": [
        {"name": "HI-BLACK 50L (5%)", "desc": "카본블랙 5% 희석 마스터배치 (명도/어두움 제어)"},
        {"name": "P-261 YELLOW", "desc": "황색 유기 안료 (노란기 b* 축 제어)"},
        {"name": "RED-107 (5%)", "desc": "적색 안료 5% 희석액 (붉은기 a* 축 미세 보정)"},
        {"name": "GREEN L8710 (5%)", "desc": "프탈로시아닌 그린 5% 희석액 (녹색기 a* 축 제어)"},
        {"name": "Phthalo Blue (5%)", "desc": "프탈로시아닌 블루 5% 희석 마스터배치 (푸른기 제어)"},
        {"name": "Violet 23 (5%)", "desc": "바이올렛 유기안료 5% 희석액 (보라/적청기 조율)"}
    ]
}

def load_library():
    if os.path.exists(LIBRARY_FILE):
        try:
            with open(LIBRARY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_LIBRARY
    return DEFAULT_LIBRARY

def save_library(lib_data):
    with open(LIBRARY_FILE, "w", encoding="utf-8") as f:
        json.dump(lib_data, f, ensure_ascii=False, indent=2)

if "library" not in st.session_state:
    st.session_state.library = load_library()

if "materials" not in st.session_state:
    st.session_state.materials = {
        "cat1": [
            {"name": "PEP-1681", "desc": "폴리에스터 수지 (주 바인더 수지계)", "amount": 360.00},
            {"name": "E-12", "desc": "에폭시 수지 (경화 및 도막 내화학성 부여)", "amount": 240.00},
            {"name": "HM3328", "desc": "경화제 및 레벨링 조절 보조제", "amount": 14.00}
        ],
        "cat2": [
            {"name": "LA-100", "desc": "유동성 향상 및 대전 특성제어 첨가제", "amount": 10.00},
            {"name": "BENZOIN", "desc": "도막 소성 시 가스 기포 방출 탈포제", "amount": 3.00},
            {"name": "PA-103", "desc": "표면 오렌지필 방지 레벨링 조정제", "amount": 3.00}
        ],
        "cat3": [
            {"name": "TiO2 R-996", "desc": "이산화티타늄 고은폐 백색 안료 (빛 반사율 제어)", "amount": 285.00},
            {"name": "CaCO3", "desc": "탄산칼슘 체질재 (도막 스페이스 및 소광/비용 조절)", "amount": 75.29}
        ],
        "cat4": [
            {"name": "HI-BLACK 50L (5%)", "desc": "카본블랙 5% 희석 마스터배치 (명도/어두움 제어)", "amount": 0.49},
            {"name": "P-261 YELLOW", "desc": "황색 유기 안료 (노란기 b* 축 제어)", "amount": 6.63},
            {"name": "RED-107 (5%)", "desc": "적색 안료 5% 희석액 (붉은기 a* 축 미세 보정)", "amount": 1.94},
            {"name": "GREEN L8710 (5%)", "desc": "프탈로시아닌 그린 5% 희석액 (녹색기 a* 축 제어)", "amount": 0.00}
        ]
    }

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def explain_material_with_ai(mat_name, api_key):
    if not api_key:
        return "API Key 미입력 상태 (Gemini API Key 필요)"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""
        분체도장(Powder Coating) 및 도료 화학 관점에서 원료/안료 '{mat_name}'의 핵심 역할과 특징을 1문장으로 간결하고 전문적으로 설명해줘.
        """
        response = model.generate_content(prompt)
        return response.text.strip().replace("\n", " ")
    except Exception as e:
        return f"AI 분석 실패: {e}"

# 사이드바 설정
with st.sidebar:
    st.title("⚙️ 시스템 설정")
    api_key = st.text_input("🔑 Gemini API Key", type="password")
    st.markdown("[👉 API Key 무료 발급받기](https://aistudio.google.com/app/apikey)")
    
    if api_key:
        st.success("🟢 AI 실시간 연동 완료")
    else:
        st.warning("🔴 API Key를 입력해주세요")
        
    st.divider()
    st.markdown("### 🏢 조직 및 제작자")
    st.caption("**소속:** DYP주식회사 기술연구소")
    st.caption("**담당:** 최새한 대리")
    
    st.divider()
    st.markdown("### 🛠️ 데이터 관리")
    col_reset1, col_reset2 = st.columns(2)
    with col_reset1:
        if st.button("🗑️ 재료 DB 리셋", use_container_width=True):
            if os.path.exists(LIBRARY_FILE):
                os.remove(LIBRARY_FILE)
            st.session_state.library = DEFAULT_LIBRARY
            st.success("재료 DB가 기본값으로 초기화되었습니다.")
            st.rerun()
    with col_reset2:
        if st.button("💬 Q&A 리셋", use_container_width=True):
            st.session_state.chat_history = []
            st.success("대화 내역이 리셋되었습니다.")
            st.rerun()

# 1. 목표 색상 (STD Target)
with st.container(border=True):
    st.markdown("#### 🎯 1. 목표 색상 (STD Target)")
    c1, c2, c3 = st.columns(3)
    with c1:
        std_l = st.number_input("STD L* (명도)", value=89.63, format="%.2f")
    with c2:
        std_a = st.number_input("STD a* (적/녹)", value=1.32, format="%.2f")
    with c3:
        std_b = st.number_input("STD b* (황/청)", value=11.19, format="%.2f")

# 2. 전체 배합 (한 줄에 2개씩 2x2 그리드)
current_formula = {}

def render_category(cat_key, cat_title, cat_css_class, cat_desc):
    st.markdown(f'<div class="cat-header {cat_css_class}"><span>{cat_title}</span></div>', unsafe_allow_html=True)
    st.caption(cat_desc)
    
    items_to_remove = []
    for idx, item in enumerate(st.session_state.materials[cat_key]):
        c_name, c_amt, c_del = st.columns([2.2, 1.8, 1])
        with c_name:
            item["name"] = st.text_input(f"재료명 #{idx+1}", value=item["name"], key=f"name_{cat_key}_{idx}")
        with c_amt:
            item["amount"] = st.number_input(f"투입량(g) #{idx+1}", value=float(item["amount"]), format="%.2f", step=0.01, key=f"amt_{cat_key}_{idx}")
        with c_del:
            st.write("")
            if st.button("🗑️", key=f"del_{cat_key}_{idx}"):
                items_to_remove.append(idx)
        
        st.markdown(f"""
        <div class="ai-info-box">
            💡 <b>AI 역할:</b> {item.get('desc', 'AI 분석 중...')}
        </div>
        """, unsafe_allow_html=True)
        
        current_formula[item["name"]] = item["amount"]
        
    if items_to_remove:
        for i in sorted(items_to_remove, reverse=True):
            st.session_state.materials[cat_key].pop(i)
        st.rerun()

    cat_short = cat_title.split(']')[1] if ']' in cat_title else cat_title
    with st.expander(f"➕ {cat_short} 재료 추가", expanded=False):
        st.markdown("**방법 1. 기존 DB 선택**")
        lib_items = st.session_state.library.get(cat_key, [])
        lib_names = [item["name"] for item in lib_items]
        selected_lib_name = st.selectbox(f"DB 목록 선택 ({cat_short})", options=["-- 선택하세요 --"] + lib_names, key=f"sel_lib_{cat_key}")
        init_amt_lib = st.number_input("투입량(g)", value=1.00, format="%.2f", step=0.01, key=f"amt_lib_{cat_key}")
        if st.button("➕ DB 재료 추가", key=f"btn_add_lib_{cat_key}", use_container_width=True):
            if selected_lib_name != "-- 선택하세요 --":
                lib_match = next((x for x in lib_items if x["name"] == selected_lib_name), None)
                desc_val = lib_match["desc"] if lib_match else "저장된 원료"
                st.session_state.materials[cat_key].append({
                    "name": selected_lib_name,
                    "desc": desc_val,
                    "amount": init_amt_lib
                })
                st.success(f"'{selected_lib_name}' 추가 완료!")
                st.rerun()
            else:
                st.warning("목록에서 재료를 선택해주세요.")

        st.markdown("**방법 2. 직접 입력 (AI 분석)**")
        typed_name = st.text_input(f"신규 {cat_short} 재료명", placeholder="예: Phthalo Blue 5%...", key=f"type_name_{cat_key}")
        init_amt_typed = st.number_input("투입량(g)", value=1.00, format="%.2f", step=0.01, key=f"amt_typed_{cat_key}")
        if st.button("➕ 신규 입력 & DB 저장", key=f"btn_add_typed_{cat_key}", use_container_width=True):
            if typed_name.strip():
                name_clean = typed_name.strip()
                with st.spinner("🧠 AI 역할 분석 중..."):
                    ai_desc = explain_material_with_ai(name_clean, api_key)
                
                st.session_state.materials[cat_key].append({
                    "name": name_clean,
                    "desc": ai_desc,
                    "amount": init_amt_typed
                })
                
                if not any(x["name"] == name_clean for x in st.session_state.library[cat_key]):
                    st.session_state.library[cat_key].append({
                        "name": name_clean,
                        "desc": ai_desc
                    })
                    save_library(st.session_state.library)
                
                st.success(f"'{name_clean}' 추가 및 DB 저장 완료!")
                st.rerun()
            else:
                st.warning("재료명을 입력해주세요.")

with st.container(border=True):
    st.markdown("#### 🧪 2. 전체 배합")
    
    col_row1_1, col_row1_2 = st.columns(2)
    with col_row1_1:
        render_category("cat1", "🏢 [카테고리 1] 수지 & 경화제", "cat-1-bg", "도막의 기본 바인더 및 경화 반응 성분")
    with col_row1_2:
        render_category("cat2", "⚡ [카테고리 2] 기능성 첨가제", "cat-2-bg", "레벨링, 탈포, 슬립 제어 성분")
        
    st.divider()
    
    col_row2_1, col_row2_2 = st.columns(2)
    with col_row2_1:
        render_category("cat3", "⚪ [카테고리 3] 은폐 & 체질 안료", "cat-3-bg", "은폐력, 은폐층 형성 및 물성 조절 성분")
    with col_row2_2:
        render_category("cat4", "🎨 [카테고리 4] 유색 착색 안료", "cat-4-bg", "L*, a*, b* 색상 미세 조율용 안료")

# 전체 배합 중량 KPI 대시보드
total_weight = sum(current_formula.values())
st.markdown(f"""
<div class="kpi-container">
    <div>
        <div class="kpi-label">전체 설정 처방 총중량 (Total Batch Weight)</div>
        <div style="font-size:0.82rem; color:#64748B;">총 {len(current_formula)}개 등록 재료 성분 밸런스 합산</div>
    </div>
    <div class="kpi-val" style="color:#2563EB;">{total_weight:.2f} <span style="font-size:1.1rem; color:#64748B;">g</span></div>
</div>
""", unsafe_allow_html=True)

# 3. 차수 및 CIE 오차
with st.container(border=True):
    st.markdown("#### 📐 3. 차수 및 측정 결과 (CIE 오차 Δ)")
    c_b, c_l, c_a, c_b_axis = st.columns([1, 1, 1, 1])
    with c_b:
        batch_num = st.number_input("현재 배치 차수 (Batch No.)", min_value=1, value=8)
    with c_l:
        dl = st.number_input("Δ L* (명도 오차)", value=0.28, format="%.2f", step=0.01)
    with c_a:
        da = st.number_input("Δ a* (적/녹 오차)", value=-0.15, format="%.2f", step=0.01)
    with c_b_axis:
        db = st.number_input("Δ b* (황/청 오차)", value=0.10, format="%.2f", step=0.01)

st.divider()
calc_btn = st.button("🚀 동적 전체 처방 & AI 초정밀 0.01g 보정 산출", type="primary", use_container_width=True)

if calc_btn:
    if not api_key:
        st.error("⚠️ 사이드바에 Gemini API Key를 먼저 입력해 주세요.")
    else:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')

            formula_str = "\n".join([f"- {k}: {v:.2f}g" for k, v in current_formula.items()])

            prompt = f"""
            너는 DYP주식회사 기술연구소의 최고급 추론 능력을 가진 초정밀 CCM 안료 보정 AI 전문가야.
            사용자가 동적으로 설정한 아래의 전체 처방 구조(동적 재료 목록)와 목표 색상(STD), 현재 측정 오차를 종합 분석하여,
            각 안료/첨가제 성분의 비선형 포화, 은폐/체질재에 따른 Wash-out 효과, 5% 희석 마스터배치 민감도를 반영한 다음 차수({batch_num + 1}번) 0.01g 초정밀 처방을 작성해줘.

            [DYP 기술연구소 보안 주의사항]
            본 데이터 및 처방 정보는 DYP주식회사 영업비밀에 해당하므로 대외 보안을 준수하여 작성한다.

            [목표 색상 STD]
            L*: {std_l}, a*: {std_a}, b*: {std_b}

            [현재 사용자가 설정한 동적 처방 구조 (총 {total_weight:.2f}g)]
            {formula_str}

            [가장 최근 {batch_num}번 측정 오차]
            - dL: {dl}, da: {da}, db: {db}

            다음 양식으로 가독성 높은 마크다운 형태로 답변해줘:
            ### 1. 🔍 처방 성분 분석 및 오차 발생 원인
            (사용자가 설정한 수지, 은폐재, 착색 안료 간의 흡광/산란 물리적 반응 분석)

            ### 2. 🎯 {batch_num + 1}번 0.01g 초정밀 배합 처방
            (기존에 등록된 동적 재료별 변경량 및 최종 목표 투입g 제시)

            ### 3. 💡 보정 근거 및 0.01g 산출 이유
            (각 성분 조정량에 대한 화학적/물리적 근거 명확히 제시)
            """

            with st.spinner("🧠 AI가 동적 처방 및 성분 간 빛 산란 매커니즘을 학습하여 0.01g 처방을 산출 중입니다..."):
                response = model.generate_content(prompt)
                st.markdown("---")
                with st.container(border=True):
                    st.markdown(f"### 📋 DYP 기술연구소 AI 초정밀 분석 보고서 (Batch #{batch_num + 1})")
                    st.markdown(response.text)

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")

# AI 실시간 Q&A 대화창
st.divider()
with st.container(border=True):
    st.markdown("#### 💬 DYP AI 실시간 기술 상담 & 대화창 (Gemini Pro Engine)")
    st.caption("신규 추가한 재료의 특성, 조색 원인, 분체도장 물성 등 자유롭게 질의하세요. (최고 수준 추론 모드)")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("질문을 입력하세요"):
        if not api_key:
            st.error("⚠️ 좌측 사이드바에 Gemini API Key를 먼저 입력해 주세요.")
        else:
            st.chat_message("user").markdown(user_query)
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')

                formula_summary = ", ".join([f"{k}({v}g)" for k, v in current_formula.items()])

                system_context = f"""
                너는 DYP주식회사 기술연구소의 최고 수석 연구원급 AI 분석가야.
                상대방은 기술연구소 최새한 대리 및 동료 연구원이야.
                질문자의 의도를 깊이 있게 파악하여 물리 화학적 원리, 분체도장 도막 형성 메커니즘, 안료 입자 분산, 쿠벨카-뭉크 이론, 
                그리고 파스텔 베이스 Wash-out 효과 등에 기반한 최고 수준의 논리적이고 전문적인 답변(Max Reasoning)을 제공해줘.

                [현재 사용자의 실시간 배합 설정]
                - STD 타겟: L* {std_l}, a* {std_a}, b* {std_b}
                - 현재 동적 배합 ({total_weight:.2f}g): {formula_summary}
                - 현재 오차: dL {dl}, da {da}, db {db}

                질문: {user_query}
                """

                with st.chat_message("assistant"):
                    with st.spinner("🧠 최고 사고 수준(Max Reasoning)으로 깊이 있게 분석 중입니다..."):
                        response = model.generate_content(system_context)
                        st.markdown(response.text)
                        st.session_state.chat_history.append({"role": "assistant", "content": response.text})

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

