"""
Utility functions and shared theme for BuzNet Major Project
B.Tech Final Year Project - Business Intelligence System
Premium Blue & White SaaS Theme — v4.0
Primary: #2563EB | Secondary: #3B82F6 | BG: #F8FAFC | Card: #FFFFFF | Text: #1E293B
"""
import streamlit as st
import pandas as pd
from supabase import create_client, Client

# ─────────────────────────────────────────────────────────────────────────────
# PREMIUM BLUE & WHITE SAAS THEME — v4.0
# ─────────────────────────────────────────────────────────────────────────────
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:ital,opsz,wght@0,14..32,400;0,14..32,500;0,14..32,600&display=swap');

/* ── Root Variables ── */
:root {
    --primary:       #2563EB;
    --primary-dark:  #1D4ED8;
    --primary-light: #3B82F6;
    --primary-soft:  #EFF6FF;
    --primary-mist:  #DBEAFE;
    --bg:            #F8FAFC;
    --bg-alt:        #F1F5F9;
    --card:          #FFFFFF;
    --text:          #1E293B;
    --text-sec:      #475569;
    --muted:         #64748B;
    --border:        #E2E8F0;
    --border-focus:  #93C5FD;
    --shadow-xs:     0 1px 3px rgba(37,99,235,0.06),0 1px 2px rgba(0,0,0,0.04);
    --shadow-sm:     0 2px 8px rgba(37,99,235,0.08),0 1px 3px rgba(0,0,0,0.05);
    --shadow-md:     0 4px 16px rgba(37,99,235,0.12),0 2px 6px rgba(0,0,0,0.05);
    --shadow-lg:     0 8px 32px rgba(37,99,235,0.16),0 4px 12px rgba(0,0,0,0.06);
    --success:       #10B981;
    --success-soft:  #ECFDF5;
    --danger:        #EF4444;
    --danger-soft:   #FEF2F2;
    --warning:       #F59E0B;
    --warning-soft:  #FFFBEB;
    --radius-sm:     8px;
    --radius-md:     12px;
    --radius-lg:     16px;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Poppins', 'Inter', sans-serif !important;
}
.stApp { background: var(--bg) !important; }
.main .block-container {
    padding: 1.8rem 2.4rem 3rem !important;
    max-width: 1380px !important;
}
h1,h2,h3,h4,h5,h6 {
    font-family: 'Poppins', sans-serif !important;
    color: var(--text) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 2px 0 12px rgba(37,99,235,0.06) !important;
}
[data-testid="stSidebar"]::before {
    content: '';
    display: block;
    height: 3px;
    background: linear-gradient(90deg, var(--primary) 0%, var(--primary-light) 100%);
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--muted) !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
}
[data-testid="stSidebarNav"] a {
    border-radius: var(--radius-sm) !important;
    transition: all 0.18s ease !important;
    font-weight: 500 !important;
    font-size: 0.855rem !important;
    color: var(--muted) !important;
    margin-bottom: 2px !important;
}
[data-testid="stSidebarNav"] a:hover {
    background: var(--primary-soft) !important;
    color: var(--primary) !important;
    transform: translateX(2px) !important;
}
[data-testid="stSidebarNav"] a[aria-selected="true"] {
    background: var(--primary-soft) !important;
    color: var(--primary) !important;
    font-weight: 600 !important;
    box-shadow: inset 3px 0 0 var(--primary) !important;
}

/* ── Primary Buttons ── */
.stButton > button {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.15px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 6px rgba(37,99,235,0.28) !important;
    font-family: 'Poppins', sans-serif !important;
}
.stButton > button:hover {
    background: var(--primary-dark) !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.38) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}
.stButton > button:focus {
    box-shadow: 0 0 0 3px rgba(37,99,235,0.22) !important;
    outline: none !important;
}
.stDownloadButton > button {
    background: var(--success) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    box-shadow: 0 2px 6px rgba(16,185,129,0.28) !important;
    transition: all 0.2s !important;
    font-family: 'Poppins', sans-serif !important;
}
.stDownloadButton > button:hover {
    background: #059669 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(16,185,129,0.35) !important;
}

/* ── Form Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stDateInput > div > div > input,
.stTextArea > div > div > textarea {
    border-radius: var(--radius-sm) !important;
    border: 1.5px solid var(--border) !important;
    background: #FAFCFF !important;
    color: var(--text) !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.875rem !important;
    transition: all 0.18s ease !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stDateInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
    background: white !important;
    outline: none !important;
}
.stSelectbox > div > div,
.stMultiSelect > div > div {
    border-radius: var(--radius-sm) !important;
    border: 1.5px solid var(--border) !important;
    background: #FAFCFF !important;
    font-size: 0.875rem !important;
    transition: all 0.18s ease !important;
}
.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}
.stTextInput label, .stNumberInput label, .stSelectbox label,
.stDateInput label, .stTextArea label, .stMultiSelect label,
.stSlider label, .stRadio label, .stCheckbox label {
    font-size: 0.77rem !important;
    font-weight: 600 !important;
    color: var(--text-sec) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    font-family: 'Poppins', sans-serif !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem !important;
    background: var(--bg-alt) !important;
    padding: 0.3rem !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    padding: 0.5rem 1.2rem !important;
    font-weight: 500 !important;
    font-size: 0.845rem !important;
    color: var(--muted) !important;
    transition: all 0.18s ease !important;
    background: transparent !important;
    font-family: 'Poppins', sans-serif !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: white !important;
    color: var(--primary) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--primary) !important;
    color: white !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.28) !important;
    font-weight: 600 !important;
}

/* ── Alerts ── */
.stSuccess {
    background: var(--success-soft) !important;
    border-left: 4px solid var(--success) !important;
    border-radius: var(--radius-sm) !important;
    border-top: none !important; border-right: none !important; border-bottom: none !important;
}
.stError {
    background: var(--danger-soft) !important;
    border-left: 4px solid var(--danger) !important;
    border-radius: var(--radius-sm) !important;
    border-top: none !important; border-right: none !important; border-bottom: none !important;
}
.stWarning {
    background: var(--warning-soft) !important;
    border-left: 4px solid var(--warning) !important;
    border-radius: var(--radius-sm) !important;
    border-top: none !important; border-right: none !important; border-bottom: none !important;
}
.stInfo {
    background: var(--primary-soft) !important;
    border-left: 4px solid var(--primary) !important;
    border-radius: var(--radius-sm) !important;
    border-top: none !important; border-right: none !important; border-bottom: none !important;
}

/* ── DataFrames ── */
.stDataFrame {
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
    border: 1px solid var(--border) !important;
}
.stDataFrame thead tr th {
    background: var(--primary-soft) !important;
    color: var(--primary) !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
}
.stDataFrame tbody tr:hover td {
    background: var(--primary-soft) !important;
}

/* ── Progress ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--primary), var(--primary-light)) !important;
    border-radius: 6px !important;
}
.stProgress > div > div {
    background: var(--bg-alt) !important;
    border-radius: 6px !important;
}

/* ── Metrics ── */
[data-testid="stMetricValue"] {
    color: var(--primary) !important;
    font-weight: 800 !important;
    font-size: 1.55rem !important;
    font-family: 'Poppins', sans-serif !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    height: 1px !important;
    margin: 1.5rem 0 !important;
    background: linear-gradient(90deg,transparent,var(--border),transparent) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: white !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    font-size: 0.875rem !important;
    font-family: 'Poppins', sans-serif !important;
    transition: all 0.18s !important;
}
.streamlit-expanderHeader:hover {
    border-color: var(--primary) !important;
    color: var(--primary) !important;
    background: var(--primary-soft) !important;
}
.streamlit-expanderContent {
    border: 1.5px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
    background: white !important;
}

/* ── Toggle / Checkbox ── */
.stCheckbox label, .stToggle label {
    font-weight: 500 !important;
    color: var(--text) !important;
    font-size: 0.875rem !important;
}

/* ── Radio ── */
.stRadio > div { gap: 0.45rem !important; }
.stRadio > div > label {
    background: white !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.38rem 0.9rem !important;
    font-size: 0.845rem !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
    cursor: pointer !important;
    font-family: 'Poppins', sans-serif !important;
}
.stRadio > div > label:hover {
    border-color: var(--primary) !important;
    background: var(--primary-soft) !important;
    color: var(--primary) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: white !important;
    border: 2px dashed #93C5FD !important;
    border-radius: var(--radius-md) !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--primary) !important;
    background: var(--primary-soft) !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, var(--primary), var(--primary-light)) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--primary-mist); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary-light); }

/* ══════════════════════════════════════════
   CUSTOM BuzNet COMPONENTS
══════════════════════════════════════════ */

/* ── Card ── */
.bz-card {
    background: var(--card);
    border-radius: var(--radius-md);
    padding: 1.4rem 1.6rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
    transition: box-shadow 0.2s ease;
}
.bz-card:hover { box-shadow: var(--shadow-md); }

/* ── Page Header Banner ── */
.bz-header {
    background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 55%, #3B82F6 100%);
    border-radius: var(--radius-lg);
    padding: 2rem 2.5rem;
    color: white;
    text-align: center;
    box-shadow: var(--shadow-lg);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.bz-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -40px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
    pointer-events: none;
}
.bz-header::after {
    content: '';
    position: absolute;
    bottom: -60px; left: -30px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
    pointer-events: none;
}
.bz-header h1 {
    font-size: 1.85rem;
    font-weight: 800;
    margin: 0 0 0.35rem;
    letter-spacing: -0.4px;
    font-family: 'Poppins', sans-serif;
    color: white !important;
    position: relative;
    z-index: 1;
}
.bz-header p {
    font-size: 0.92rem;
    opacity: 0.88;
    margin: 0;
    font-weight: 400;
    position: relative;
    z-index: 1;
}

/* ── KPI Card ── */
.bz-kpi {
    background: var(--card);
    border-radius: var(--radius-md);
    padding: 1.3rem 1.5rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
    border-top: 3px solid var(--primary);
    transition: all 0.22s ease;
    position: relative;
    overflow: hidden;
}
.bz-kpi::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 60px; height: 60px;
    border-radius: 0 0 0 100%;
    background: var(--primary-soft);
    opacity: 0.7;
}
.bz-kpi:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-3px);
    border-top-color: var(--primary-dark);
}
.bz-kpi .icon  { font-size: 1.5rem; margin-bottom: 0.45rem; display: block; position: relative; z-index: 1; }
.bz-kpi .label {
    font-size: 0.7rem; color: var(--muted); font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.7px; margin-bottom: 0.35rem;
    font-family: 'Poppins', sans-serif;
}
.bz-kpi .value {
    font-size: 1.5rem; font-weight: 800; color: var(--text); line-height: 1.2;
    font-family: 'Poppins', sans-serif; position: relative; z-index: 1;
}
.bz-kpi .delta {
    font-size: 0.74rem; color: var(--success); font-weight: 600; margin-top: 0.35rem;
    font-family: 'Poppins', sans-serif;
}

/* ── Section Title ── */
.bz-section-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    margin: 1.8rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0 0 0 0.75rem;
    border-left: 3px solid var(--primary);
    font-family: 'Poppins', sans-serif;
    letter-spacing: -0.1px;
}

/* ── Badge ── */
.bz-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    background: var(--primary-soft);
    color: var(--primary);
    border: 1px solid var(--primary-mist);
    font-family: 'Poppins', sans-serif;
}

/* ── Insight row ── */
.bz-insight {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.9rem 1.1rem;
    border-radius: var(--radius-sm);
    margin-bottom: 0.6rem;
    border-left: 4px solid;
    font-size: 0.875rem;
    line-height: 1.6;
    background: white;
    box-shadow: var(--shadow-xs);
    font-family: 'Poppins', sans-serif;
}
</style>
"""

def inject_theme():
    st.markdown(THEME_CSS, unsafe_allow_html=True)

def check_login():
    if not st.session_state.get("logged_in"):
        inject_theme()
        st.markdown("""
        <div class="bz-card" style="text-align:center;padding:3rem;max-width:460px;margin:5rem auto;">
            <div style="width:56px;height:56px;background:#EFF6FF;border:2px solid #DBEAFE;
                border-radius:14px;display:flex;align-items:center;justify-content:center;
                margin:0 auto 1.2rem;font-size:1.6rem;">🔒</div>
            <h2 style="color:#1E293B;margin-bottom:0.4rem;font-size:1.25rem;font-weight:700;">
                Authentication Required</h2>
            <p style="color:#64748B;font-size:0.875rem;line-height:1.6;">
                Please sign in to access this module.</p>
        </div>""", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔑 Go to Sign In", width='stretch'):
                st.switch_page("pages/0_Auth.py")
        st.stop()

@st.cache_resource
def init_connection() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

def load_user_data():
    try:
        supabase = init_connection()
        user = st.session_state.get("username", "")
        res = supabase.table("buznet_data").select("*").eq("client_id", user).execute()
        if res.data:
            df = pd.DataFrame(res.data)
            df['Date'] = pd.to_datetime(df['Date'])
            for col in ['Production', 'Sold', 'Revenue']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def format_currency(amount, symbol=None):
    sym = symbol or st.session_state.get("currency_symbol", "₹")
    return f"{sym}{amount:,.2f}"

def page_header(icon, title, subtitle):
    st.markdown(f"""
    <div class="bz-header">
        <h1>{icon} {title}</h1>
        <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)

def kpi_card(icon, label, value, delta=None):
    delta_html = f'<div class="delta">▲ {delta}</div>' if delta else ""
    return f"""
    <div class="bz-kpi">
        <div class="icon">{icon}</div>
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {delta_html}
    </div>"""
