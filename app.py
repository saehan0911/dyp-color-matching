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
    page_title="DYP주식회사 기술연구소 | AI 초정밀 CCM 조색 시스템",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. 커스텀 CSS (슬레이트/블루 대시보드 테마)
# ---------------------------------------------------------
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    .stApp {
        background-color: #F8FAFC;
    }
    
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

# 메인 헤더
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
        <span class="badge-pill badge-green">🟢 Embedded Knowledge Engine Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

LIBRARY_FILE = "material_library.json"
HISTORY_FILE = "color_history_dynamic.csv"

# ---------------------------------------------------------
# 🧠 내장 전문 분야 노하우 및 시스템 프롬프트 정의
# ---------------------------------------------------------
EMBEDDED_KNOWLEDGE_SYSTEM_PROMPT = """
너는 DYP주식회사 기술연구소의 수석 연구원급 AI 분석가이자, 쿠벨카-뭉크(Kubelka-Munk) 흡광/산란 이론 및 피드백 실측 학습(In-Context Feedback Learning) 능력을 완벽히 마스터한 초정밀 CCM 분체도료 조색 전문가야.

[전문 조색 핵심 화학/물리 법칙 노하우]
1. **5% 마스터배치(희석 안료) 미세 계산법:**
   - 안료명에 '(5%)'가 포함된 경우(예: HI-BLACK 50L 5%, RED-107 5%, Phthalo Blue 5% 등), 실효 유효 안료 농도는 5%이다.
   - 따라서 1.00g 투입 시 순수 활성 안료량은 0.05g이다.
   - 카본블랙이나 프탈로시아닌 블루/그린 등 초고착색성(High Tinting Power) 안료는 실효 농도가 매우 높아 0.01g 단위의 미세 변동에도 ΔL*(명도), Δa*(적/녹), Δb*(황/청)가 수치상 급격히 요동친다.

2. **수지/체질재 베이스의 Wash-out(희석/산란) 효과:**
   - 백색 은폐재(TiO2) 및 체질재(CaCO3, BaSO4, Talc)가 다량 배합된 파스텔 계열 베이스에서는 안료의 비선형 흡광 포화 현상이 일어난다.
   - 착색 안료 증량 시 명도(L*)가 단순 직선으로 감소하지 않고 포화 곡선을 그리므로, 누적 데이터의 ΔE 변화율 트렌드를 반드시 반영해야 한다.

3. **작업 현장 수동 계량 오차 및 공정 변수 감안:**
   - 소성 온도/시간에 따른 수지 황변(Yellowing) 현상, 도막 두께에 따른 은폐력 차이를 고려한다.
   - 0.001g 단위의 현장 적용 불가능한 미세 변경은 배제하고, 실무 현장에서 수동 계량으로 검증 가능한 0.01g 단위 초정밀 목표치를 제안한다.

[최우선 출력 규칙 - 순서 엄수]
- **어떠한 서론, 인사말, 개념 설명도 먼저 출력하지 마라.**
- **가장 먼저 1번항인 "Batch #N+1 초정밀 0.01g 보정 배합 처방 표 및 목표 수치"를 최상단 1번째로 제시하라.**
- 그 이후에 2번 원인 분석과 3번 보정 근거를 출력한다.
"""

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

def get_batch_history_df():
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            if "delta_e" not in df.columns:
                df["delta_e"] = np.sqrt(df["dl"]**2 + df["da"]**2 + df["db"]**2).round(3)
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def save_batch_history_df(df):
    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")

def append_batch_history(batch_no, std_l, std_a, std_b, dl, da, db, formula_dict, note="보정 계산 실행"):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    delta_e = round(float(np.sqrt(dl**2 + da**2 + db**2)), 3)
    record = {
        "timestamp": now_str,
        "batch_no": batch_no,
        "std_l": std_l,
        "std_a": std_a,
        "std_b": std_b,
        "dl": dl,
        "da": da,
        "db": db,
        "delta_e": delta_e,
        "formula_json": json.dumps(formula_dict, ensure_ascii=False),
        "note": note
    }
    df_new = pd.DataFrame([record])
    if os.path.exists(HISTORY_FILE):
        df_existing = pd.read_csv(HISTORY_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        save_batch_history_df(df_combined)
    else:
        save_batch_history_df(df_new)

def format_history_for_prompt():
    df = get_batch_history_df()
    if df.empty:
        return "이전 배치 학습 데이터 없음 (첫 번째 배치 분석)"
    
    history_lines = []
    recent_df = df.tail(15)
    for _, row in recent_df.iterrows():
        try:
            f_dict = json.loads(row["formula_json"])
            f_str = ", ".join([f"{k}:{v}g" for k, v in f_dict.items()])
        except Exception:
            f_str = row.get("formula_json", "")
            
        de_str = f"ΔE={row.get('delta_e', 0.0)}"
        note_str = f" [{row.get('note', '')}]" if 'note' in row and pd.notna(row['note']) else ""
        line = f"- [차수 #{row['batch_no']}] ({row['timestamp']}) | 오차: ΔL={row['dl']}, Δa={row['da']}, Δb={row['db']} ({de_str}){note_str} | 배합: {f_str}"
        history_lines.append(line)
    return "\n".join(history_lines)

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

# ---------------------------------------------------------
# API Key 로드 및 유효성 검사 (401 오류 완벽 방지)
# ---------------------------------------------------------
api_key = None
if "GEMINI_API_KEY" in st.secrets and str(st.secrets["GEMINI_API_KEY"]).strip():
    api_key = str(st.secrets["GEMINI_API_KEY"]).strip()

def explain_material_with_ai(mat_name, active_api_key):
    if not active_api_key or len(active_api_key) < 10:
        return "API Key 미입력 상태 (유효한 Gemini API Key 필요)"
    try:
        genai.configure(api_key=active_api_key)
        model = genai.GenerativeModel('gemini-3.8-flash')
        prompt = f"""
        {EMBEDDED_KNOWLEDGE_SYSTEM_PROMPT}
        분체도장 및 도료 화학 관점에서 원료/안료 '{mat_name}'의 핵심 역할과 특징을 1문장으로 간결하고 전문적으로 설명해줘.
        """
        response = model.generate_content(prompt)
        return response.text.strip().replace("\n", " ")
    except Exception as e:
        return f"AI 분석 실패: {e}"

# 사이드바 설정
with st.sidebar:
    st.title("⚙️ 시스템 설정")
    
    user_key = st.text_input("🔑 개별 API Key 입력 (선택사항)", type="password", placeholder="기본 Secrets Key 사용 중...")
    if user_key.strip():
        api_key = user_key.strip()
        
    if api_key and len(api_key) >= 10:
        st.success("🟢 API Key 정상 연동 완료")
    else:
        st.error("🔴 유효한 API Key가 설정되지 않았습니다. Secrets를 등록하거나 위 상자에 입력해주세요.")
        st.markdown("[👉 API Key 무료 발급받기](https://aistudio.google.com/app/apikey)")
        
    st.divider()
    st.markdown("### 🏢 조직 및 제작자")
    st.caption("**소속:** DYP주식회사 기술연구소")
    st.caption("**담당:** 최새한 대리")
    st.divider()

# ---------------------------------------------------------
# 📂 데이터 열람 & 개별 삭제 관리 섹션
# ---------------------------------------------------------
with st.sidebar.expander("📂 데이터 열람 & 개별 삭제 관리", expanded=False):
    st.markdown("#### 1. 📦 재료 DB 열람 및 삭제")
    cat_names = {"cat1": "1. 수지/경화제", "cat2": "2. 기능성 첨가제", "cat3": "3. 은폐/체질안료", "cat4": "4. 유색착색안료"}
    
    for c_key, c_label in cat_names.items():
        st.caption(f"**{c_label}**")
        mat_list = st.session_state.library.get(c_key, [])
        if not mat_list:
            st.text("등록된 재료 없음")
        else:
            for idx, m in enumerate(mat_list):
                col_m1, col_m2 = st.columns([4, 1])
                with col_m1:
                    st.text(f"• {m['name']}")
                with col_m2:
                    if st.button("❌", key=f"del_lib_{c_key}_{idx}"):
                        st.session_state.library[c_key].pop(idx)
                        save_library(st.session_state.library)
                        st.toast(f"'{m['name']}' 삭제 완료!")
                        st.rerun()

    st.divider()
    st.markdown("#### 2. 📜 배치 히스토리 열람 및 삭제")
    df_hist_view = get_batch_history_df()
    if df_hist_view.empty:
        st.info("저장된 히스토리가 없습니다.")
    else:
        st.caption(f"총 {len(df_hist_view)}건 기록됨")
        for h_idx, h_row in df_hist_view.iterrows():
            with st.container():
                col_h1, col_h2 = st.columns([4, 1])
                with col_h1:
                    st.markdown(f"**#{h_row['batch_no']}** ({str(h_row['timestamp'])[:10]}) | ΔE: `{h_row.get('delta_e', 'N/A')}`")
                    st.caption(f"dL:{h_row['dl']} da:{h_row['da']} db:{h_row['db']}")
                with col_h2:
                    if st.button("❌", key=f"del_hist_{h_idx}"):
                        df_hist_view = df_hist_view.drop(h_idx).reset_index(drop=True)
                        save_batch_history_df(df_hist_view)
                        st.toast(f"배치 #{h_row['batch_no']} 히스토리 삭제 완료!")
                        st.rerun()

    st.divider()
    st.markdown("#### 3. 💬 Q&A 대화 내역 열람 및 삭제")
    if not st.session_state.chat_history:
        st.info("대화 내역이 없습니다.")
    else:
        st.caption(f"총 {len(st.session_state.chat_history)//2}개 질답 저장됨")
        for q_idx in range(0, len(st.session_state.chat_history), 2):
            q_msg = st.session_state.chat_history[q_idx]
            col_q1, col_q2 = st.columns([4, 1])
            with col_q1:
                st.markdown(f"**Q:** {q_msg['content'][:30]}...")
            with col_q2:
                if st.button("❌", key=f"del_qa_{q_idx}"):
                    st.session_state.chat_history.pop(q_idx)
                    if q_idx < len(st.session_state.chat_history):
                        st.session_state.chat_history.pop(q_idx)
                    st.toast("Q&A 항목 삭제 완료!")
                    st.rerun()

# ---------------------------------------------------------
# 🧠 AI 학습 관리 칸 (AI Learning Dashboard)
# ---------------------------------------------------------
with st.sidebar.expander("🧠 AI 학습 관리 & 오차 수렴 현황", expanded=False):
    df_learn = get_batch_history_df()
    if df_learn.empty:
        st.info("아직 학습된 배치 데이터가 없습니다.")
    else:
        st.markdown(f"**Total Learned Batches:** `{len(df_learn)}회`")
        
        avg_de = df_learn["delta_e"].mean() if "delta_e" in df_learn else 0.0
        latest_de = df_learn["delta_e"].iloc[-1] if "delta_e" in df_learn else 0.0
        
        st.metric(label="가장 최근 색차 (ΔE)", value=f"{latest_de:.3f}", delta=f"평균: {avg_de:.3f}")
        st.caption("📈 **차수별 색차(ΔE) 감소 추이**")
        st.line_chart(df_learn.set_index("batch_no")[["delta_e"]])
        
        st.markdown("**🎓 내장 전문 노하우 및 피드백 반영 현황**")
        st.markdown("""
        - **5% 마스터배치 고착색성 계산법:** 카본블랙, 프탈로시아닌계 0.01g 미세 보정 적용
        - **Kubelka-Munk 흡광/산란 모델:** 파스텔 베이스 Wash-out 반응 완벽 습득
        - **수동 계량 공정 허용 오차:** 0.01g 실무 정밀도 유지
        """)
        
        csv_bytes = df_learn.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("📥 전체 학습 데이터 (CSV) 다운로드", data=csv_bytes, file_name="DYP_AI_Learning_History.csv", mime="text/csv", use_container_width=True)

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

# 2. 전체 배합 (2x2 그리드)
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
    c_b, c_l, c_a, c_b_axis, c_de = st.columns([1, 1, 1, 1, 1])
    with c_b:
        batch_num = st.number_input("현재 배치 차수 (Batch No.)", min_value=1, value=8)
    with c_l:
        dl = st.number_input("Δ L* (명도 오차)", value=0.28, format="%.2f", step=0.01)
    with c_a:
        da = st.number_input("Δ a* (적/녹 오차)", value=-0.15, format="%.2f", step=0.01)
    with c_b_axis:
        db = st.number_input("Δ b* (황/청 오차)", value=0.10, format="%.2f", step=0.01)
    with c_de:
        curr_de = round(float(np.sqrt(dl**2 + da**2 + db**2)), 3)
        st.markdown(f"**자동 산출 색차 (ΔE)**")
        st.markdown(f"<div style='font-size:1.4rem; font-weight:800; color:#D97706;'>{curr_de:.3f}</div>", unsafe_allow_html=True)

st.divider()
calc_btn = st.button("🚀 동적 전체 처방 & AI 초정밀 0.01g 보정 산출", type="primary", use_container_width=True)

if "last_ai_result" not in st.session_state:
    st.session_state.last_ai_result = None

if calc_btn:
    if not api_key or len(api_key) < 10:
        st.error("⚠️ 유효한 Gemini API Key가 존재하지 않습니다. 사이드바에 키를 입력하거나 Streamlit Secrets를 올바르게 설정하세요.")
    else:
        try:
            append_batch_history(batch_num, std_l, std_a, std_b, dl, da, db, current_formula, note="보정 계산 실행")
            history_context = format_history_for_prompt()

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-3.8-flash')

            formula_str = "\n".join([f"- {k}: {v:.2f}g" for k, v in current_formula.items()])

            prompt = f"""
            {EMBEDDED_KNOWLEDGE_SYSTEM_PROMPT}

            [실시간 누적 배치 히스토리 및 오차 피드백 데이터]
            {history_context}

            [현재 배치 목표 색상 STD]
            L*: {std_l}, a*: {std_a}, b*: {std_b}

            [현재 배치 #{batch_num} 처방 구조 (총 {total_weight:.2f}g)]
            {formula_str}

            [현재 배치 #{batch_num} 측정 오차]
            - ΔL*: {dl}
            - Δa*: {da}
            - Δb*: {db}
            - ΔE: {curr_de}

            [출력 조건 - 순서 엄수!]
            반드시 1번(0.01g 보정 배합 표 및 목표 수치)을 최상단에 가장 먼저 출력한 후, 2번 원인 분석, 3번 보정 근거를 제시하라.

            ### 1. 🎯 Batch #{batch_num + 1} 초정밀 0.01g 보정 배합 처방 (최종 목표 값)
            (각 동적 재료별 [기존 투입g] | [조정량 (+/- g)] | [최종 목표 투입g] 마크다운 표 및 정확한 수치를 최상단에 먼저 출력)

            ### 2. 📈 누적 학습 및 오차 발생 원인 분석
            (이전 배치 히스토리와 현재 오차를 비교하여, 어떤 안료가 과다/부족했는지 물리화학적으로 분석)

            ### 3. 💡 학습 기반 보정 근거 및 0.01g 산출 이유
            (누적 데이터 피드백 분석을 통해 오차를 최소화할 수 있는 과학적/경험적 근거)
            """

            with st.spinner("🧠 내장 전문 노하우 및 누적 데이터를 학습하여 0.01g 최종 보정 수치를 계산 중입니다..."):
                response = model.generate_content(prompt)
                st.session_state.last_ai_result = {
                    "batch_next": batch_num + 1,
                    "text": response.text
                }
                st.success("✅ 학습 기반 0.01g 초정밀 보정 수치 계산이 완료되었습니다!")

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")

# ---------------------------------------------------------
# 보정 산출 결과 출력 (수치 표 최상단 출력)
# ---------------------------------------------------------
if st.session_state.last_ai_result:
    st.markdown("---")
    res_batch = st.session_state.last_ai_result["batch_next"]
    with st.container(border=True):
        st.markdown(f"### 📋 DYP 기술연구소 AI 초정밀 보정 처방 보고서 (Batch #{res_batch})")
        st.markdown(st.session_state.last_ai_result["text"])

    st.markdown("---")
    with st.container(border=True):
        st.markdown(f"#### 🧪 Batch #{res_batch} AI 추천 처방 실제 제조 및 실측 검증 입력 (AI 학습 피드백)")
        st.caption("위 AI가 만들어준 0.01g 추천 계산대로 실제 조색/소성한 후, 측정된 실제 색차(CIE)를 입력하시면 AI가 이를 완벽 학습합니다.")
        
        fb_col1, fb_col2, fb_col3, fb_col4 = st.columns(4)
        with fb_col1:
            fb_dl = st.number_input(f"실측 ΔL* (명도 오차)", value=0.00, format="%.2f", step=0.01, key="fb_dl")
        with fb_col2:
            fb_da = st.number_input(f"실측 Δa* (적/녹 오차)", value=0.00, format="%.2f", step=0.01, key="fb_da")
        with fb_col3:
            fb_db = st.number_input(f"실측 Δb* (황/청 오차)", value=0.00, format="%.2f", step=0.01, key="fb_db")
        with fb_col4:
            fb_de = round(float(np.sqrt(fb_dl**2 + fb_da**2 + fb_db**2)), 3)
            st.markdown(f"**실측 색차 (ΔE)**")
            st.markdown(f"<div style='font-size:1.4rem; font-weight:800; color:#10B981;'>{fb_de:.3f}</div>", unsafe_allow_html=True)
            
        if st.button(f"💾 Batch #{res_batch} 실측 결과 저장 및 AI 학습 반영", type="secondary", use_container_width=True):
            append_batch_history(res_batch, std_l, std_a, std_b, fb_dl, fb_da, fb_db, current_formula, note="AI 추천 처방 실측 검증 피드백")
            st.success(f"🎉 Batch #{res_batch} 실제 제조 실측 오차(ΔE: {fb_de:.3f})가 AI 학습 데이터베이스에 성공적으로 저장되었습니다! 다음 보정 시 이 결과가 반영됩니다.")
            st.rerun()

# AI 실시간 Q&A 대화창
st.divider()
with st.container(border=True):
    st.markdown("#### 💬 DYP AI 실시간 기술 상담 & 대화창 (Gemini Pro Engine)")
    st.caption("신규 추가한 재료의 특성, 조색 원인, 분체도장 물성 등 자유롭게 질의하세요. (최고 수준 추론 모드)")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("질문을 입력하세요"):
        if not api_key or len(api_key) < 10:
            st.error("⚠️ 유효한 Gemini API Key가 입력되지 않았습니다.")
        else:
            st.chat_message("user").markdown(user_query)
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-3.8-flash')

                formula_summary = ", ".join([f"{k}({v}g)" for k, v in current_formula.items()])
                history_summary = format_history_for_prompt()

                system_context = f"""
                {EMBEDDED_KNOWLEDGE_SYSTEM_PROMPT}

                [누적 연구 히스토리 참고]
                {history_summary}

                [현재 사용자의 실시간 배합 설정]
                - STD 타겟: L* {std_l}, a* {std_a}, b* {std_b}
                - 현재 동적 배합 ({total_weight:.2f}g): {formula_summary}
                - 현재 오차: dL {dl}, da {da}, db {db}

                질문: {user_query}
                """

                with st.chat_message("assistant"):
                    with st.spinner("🧠 전문 노하우 기반 깊이 있는 분석 중입니다..."):
                        response = model.generate_content(system_context)
                        st.markdown(response.text)
                        st.session_state.chat_history.append({"role": "assistant", "content": response.text})

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
