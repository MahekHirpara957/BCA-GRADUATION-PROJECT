import streamlit as st
from utils import inject_theme

st.set_page_config(
    page_title="BuzNet — Business Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_theme()

st.markdown("""
<style>
/* ── Hero ── */
.hero-wrap {
    background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 55%, #3B82F6 100%);
    border-radius: 18px;
    padding: 3.2rem 2.8rem 2.8rem;
    text-align: center;
    color: white;
    margin-bottom: 1.8rem;
    box-shadow: 0 6px 28px rgba(37,99,235,0.22);
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: ''; position: absolute; top: -80px; right: -60px;
    width: 280px; height: 280px; border-radius: 50%;
    background: rgba(255,255,255,0.06); pointer-events: none;
}
.hero-wrap::after {
    content: ''; position: absolute; bottom: -80px; left: -40px;
    width: 220px; height: 220px; border-radius: 50%;
    background: rgba(255,255,255,0.04); pointer-events: none;
}
.hero-logo {
    width: 64px; height: 64px;
    background: rgba(255,255,255,0.15);
    border: 2px solid rgba(255,255,255,0.25);
    border-radius: 18px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.8rem; margin: 0 auto 1.2rem;
    position: relative; z-index: 1;
}
.hero-title {
    font-size: 2.6rem; font-weight: 800; margin-bottom: 0.3rem;
    letter-spacing: -0.6px; font-family: 'Poppins', sans-serif;
    position: relative; z-index: 1;
}
.hero-eyebrow {
    font-size: 0.72rem; font-weight: 700; opacity: 0.7;
    text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.3rem;
    position: relative; z-index: 1;
}
.hero-sub {
    font-size: 1rem; opacity: 0.88; line-height: 1.65;
    margin-bottom: 1.4rem; font-weight: 400;
    max-width: 540px; margin-left: auto; margin-right: auto;
    position: relative; z-index: 1;
}
.hero-chips {
    display: flex; gap: 0.55rem; justify-content: center;
    flex-wrap: wrap; margin-bottom: 1.8rem;
    position: relative; z-index: 1;
}
.chip {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 999px; padding: 0.3rem 0.9rem;
    font-size: 0.78rem; font-weight: 600;
    backdrop-filter: blur(4px);
}
.stat-row {
    display: flex; gap: 0.85rem; justify-content: center;
    flex-wrap: wrap; position: relative; z-index: 1;
}
.stat-box {
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 12px; padding: 0.8rem 1.3rem;
    text-align: center; min-width: 88px;
    backdrop-filter: blur(4px);
}
.stat-box .s-val {
    font-size: 1.5rem; font-weight: 800; line-height: 1.2;
    font-family: 'Poppins', sans-serif;
}
.stat-box .s-lbl {
    font-size: 0.65rem; opacity: 0.78; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.6px; margin-top: 0.15rem;
}

/* ── Status bar ── */
.status-bar {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 10px; padding: 0.8rem 1.2rem;
    margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 0.7rem;
    box-shadow: 0 1px 4px rgba(37,99,235,0.06);
}
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #2563EB; flex-shrink: 0;
    animation: pulse-dot 1.8s infinite;
    box-shadow: 0 0 0 0 rgba(37,99,235,0.4);
}
@keyframes pulse-dot {
    0%   { box-shadow: 0 0 0 0 rgba(37,99,235,0.35); }
    70%  { box-shadow: 0 0 0 6px rgba(37,99,235,0); }
    100% { box-shadow: 0 0 0 0 rgba(37,99,235,0); }
}

/* ── Section header ── */
.section-hdr {
    display: flex; align-items: center; gap: 0.65rem;
    margin: 2rem 0 1rem;
    padding-left: 0.75rem; border-left: 3px solid #2563EB;
}
.section-hdr h2 {
    font-size: 1.1rem; font-weight: 700; color: #1E293B; margin: 0 !important;
    font-family: 'Poppins', sans-serif;
}
.section-hdr span {
    font-size: 0.72rem; color: #2563EB;
    background: #EFF6FF; border: 1px solid #DBEAFE;
    border-radius: 999px; padding: 0.15rem 0.65rem; font-weight: 600;
}

/* ── Module Grid ── */
.mod-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(255px, 1fr));
    gap: 1rem; margin-bottom: 1.8rem;
}
.mod-card {
    background: #FFFFFF; border-radius: 14px; padding: 1.4rem 1.5rem;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 6px rgba(37,99,235,0.06);
    transition: all 0.22s ease; cursor: default;
}
.mod-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(37,99,235,0.13);
    border-color: #DBEAFE;
}
.mod-icon  { font-size: 1.85rem; margin-bottom: 0.6rem; display: block; }
.mod-title { font-size: 0.96rem; font-weight: 700; color: #1E293B; margin-bottom: 0.3rem; }
.mod-desc  { font-size: 0.81rem; color: #64748B; line-height: 1.55; margin-bottom: 0.75rem; }
.mod-tags  { display: flex; gap: 0.3rem; flex-wrap: wrap; }
.mod-tag {
    background: #EFF6FF; color: #2563EB;
    border: 1px solid #DBEAFE;
    border-radius: 999px; padding: 0.14rem 0.58rem;
    font-size: 0.68rem; font-weight: 600;
}

/* ── Tech Bar ── */
.tech-bar {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px;
    padding: 1.5rem; margin: 1.4rem 0;
    display: flex; gap: 1.2rem; flex-wrap: wrap; justify-content: center;
    box-shadow: 0 1px 6px rgba(37,99,235,0.06);
}
.tech-item { text-align: center; min-width: 72px; }
.tech-item .t-icon { font-size: 1.55rem; margin-bottom: 0.25rem; }
.tech-item .t-name { font-size: 0.76rem; font-weight: 700; color: #374151; }
.tech-item .t-sub  { font-size: 0.67rem; color: #9CA3AF; }

/* ── Footer ── */
.bz-footer {
    background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 60%, #3B82F6 100%);
    border-radius: 14px;
    padding: 2rem 1.8rem;
    color: white; text-align: center; margin-top: 2.5rem;
    box-shadow: 0 4px 20px rgba(37,99,235,0.18);
}
</style>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-logo">📊</div>
    <div class="hero-eyebrow">Business Intelligence System</div>
    <div class="hero-title">BuzNet</div>
    <div class="hero-sub">
        Transform raw business data into powerful insights.<br>
        Predict the future. Understand the present. Master your market.
    </div>
    <div class="hero-chips">
        <span class="chip">📊 Real-Time Analytics</span>
        <span class="chip">🤖 Multi-Model ML</span>
        <span class="chip">☁️ Cloud-Backed</span>
        <span class="chip">🔮 AI Forecasting</span>
        <span class="chip">📥 Smart Export</span>
        <span class="chip">🔒 Secure Auth</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── STATUS BAR ────────────────────────────────────────────────────────────────
logged = st.session_state.get("logged_in", False)
user   = st.session_state.get("username", "")
if logged:
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-dot"></div>
        <span style="font-weight:600;color:#2563EB;font-size:0.85rem;">System Active</span>
        <span style="color:#64748B;font-size:0.85rem;">
            Logged in as <strong>{user}</strong> — Navigate to any module from the sidebar
        </span>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-bar" style="background:#FFFBEB;border-color:#FDE68A;">
        <div class="status-dot" style="background:#F59E0B;box-shadow:none;animation:none;"></div>
        <span style="font-weight:600;color:#D97706;font-size:0.85rem;">Not Logged In</span>
        <span style="color:#64748B;font-size:0.85rem;">
            👈 Go to <strong>Authentication</strong> from the sidebar to sign in first
        </span>
    </div>""", unsafe_allow_html=True)

# ── MODULE CARDS ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-hdr">
    <h2>🗂️ Explore Modules</h2>
    <span>Use sidebar to navigate</span>
</div>""", unsafe_allow_html=True)

MODULES = [
    {"icon":"🔒","title":"Authentication",
     "desc":"Secure login & registration powered by Supabase. JWT-based role access.",
     "tags":["Supabase","JWT","Secure"]},
    {"icon":"📝","title":"Smart Data Intake",
     "desc":"Manual entry, bulk CSV upload, auto cleaning, duplicate removal, inline edit & delete.",
     "tags":["CRUD","CSV Upload","Validation"]},
    {"icon":"📊","title":"Analytics Dashboard",
     "desc":"KPI cards, drill-down charts, revenue heatmap, Top/Bottom 5 product analysis.",
     "tags":["KPIs","Heatmap","Plotly"]},
    {"icon":"🔮","title":"ML Predictions",
     "desc":"4 ML models (LR, RF, DT, XGBoost) with auto best-model selection & multi-month forecast.",
     "tags":["XGBoost","Random Forest","Forecast"]},
    {"icon":"🔍","title":"Search & Export",
     "desc":"Advanced filters, smart search, inline edit & delete, Excel & CSV export.",
     "tags":["Excel Export","CSV","Filters"]},
    {"icon":"💡","title":"Auto Insights",
     "desc":"AI-generated business insights, alerts, trend analysis, downloadable PDF reports.",
     "tags":["AI Insights","PDF Report","Alerts"]},
    {"icon":"⚙️","title":"Settings",
     "desc":"Change password, currency selection, notification preferences, data export.",
     "tags":["Password","Currency","Notifications"]},
]

st.markdown('<div class="mod-grid">', unsafe_allow_html=True)
for m in MODULES:
    tags_html = "".join(f'<span class="mod-tag">{t}</span>' for t in m["tags"])
    st.markdown(f"""
    <div class="mod-card">
        <span class="mod-icon">{m["icon"]}</span>
        <div class="mod-title">{m["title"]}</div>
        <div class="mod-desc">{m["desc"]}</div>
        <div class="mod-tags">{tags_html}</div>
        <div style="margin-top:0.7rem;color:#2563EB;font-size:0.78rem;font-weight:600;">
            👈 Open from sidebar
        </div>
    </div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── TECH STACK ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-hdr">
    <h2>⚡ Technology Stack</h2>
    <span>BCA Major Project</span>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="tech-bar">
    <div class="tech-item"><div class="t-icon">🐍</div><div class="t-name">Python 3.11</div><div class="t-sub">Core</div></div>
    <div class="tech-item"><div class="t-icon">🌊</div><div class="t-name">Streamlit</div><div class="t-sub">Framework</div></div>
    <div class="tech-item"><div class="t-icon">🔥</div><div class="t-name">XGBoost</div><div class="t-sub">ML Engine</div></div>
    <div class="tech-item"><div class="t-icon">🤖</div><div class="t-name">Scikit-Learn</div><div class="t-sub">ML Models</div></div>
    <div class="tech-item"><div class="t-icon">📈</div><div class="t-name">Plotly</div><div class="t-sub">Charts</div></div>
    <div class="tech-item"><div class="t-icon">☁️</div><div class="t-name">Supabase</div><div class="t-sub">Database</div></div>
    <div class="tech-item"><div class="t-icon">🐼</div><div class="t-name">Pandas</div><div class="t-sub">Analysis</div></div>
    <div class="tech-item"><div class="t-icon">📄</div><div class="t-name">ReportLab</div><div class="t-sub">PDF</div></div>
    <div class="tech-item"><div class="t-icon">🌐</div><div class="t-name">Gemini AI</div><div class="t-sub">NLP</div></div>
</div>
""", unsafe_allow_html=True)

# ── PROJECT INFO ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-hdr">
    <h2>📋 Project Overview</h2>
    <span>Academic Details</span>
</div>""", unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="bz-footer">
    <div style="font-size:1.3rem;font-weight:800;margin-bottom:0.3rem;">📊 BuzNet v4.0</div>
    <div style="opacity:0.85;margin-bottom:0.6rem;font-size:0.9rem;">Business Performance Visualisation & Intelligence System</div>
</div>
""", unsafe_allow_html=True)
