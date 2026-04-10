import streamlit as st
from supabase import create_client, Client
from utils import inject_theme

st.set_page_config(page_title="Authentication — BuzNet", page_icon="🔒", layout="centered")
inject_theme()

st.markdown("""
<style>
/* ── Auth Page — Premium Blue & White ── */
.auth-wrap { max-width: 460px; margin: 0 auto; padding: 1.5rem 0; }

.auth-logo {
    text-align: center;
    margin-bottom: 2rem;
    padding: 1.8rem 1rem 1rem;
}
.auth-logo-icon {
    width: 58px; height: 58px;
    background: linear-gradient(135deg, #1D4ED8 0%, #3B82F6 100%);
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem; margin: 0 auto 0.9rem;
    box-shadow: 0 6px 20px rgba(37,99,235,0.32);
}
.auth-logo h1 {
    font-size: 1.7rem !important;
    font-weight: 800 !important;
    color: #1E293B !important;
    margin: 0 0 0.2rem !important;
    letter-spacing: -0.4px;
}
.auth-logo p { color: #64748B; font-size: 0.875rem; margin: 0; }

.auth-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 2rem 2rem 1.6rem;
    box-shadow: 0 4px 24px rgba(37,99,235,0.10), 0 1px 4px rgba(0,0,0,0.05);
    border: 1px solid #E2E8F0;
    margin-bottom: 0.75rem;
}

.auth-footer {
    text-align: center;
    color: #94A3B8;
    font-size: 0.76rem;
    padding: 1rem;
    margin-top: 0.5rem;
}
.auth-session {
    background: #EFF6FF;
    border: 1px solid #DBEAFE;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.82rem;
    color: #2563EB;
    font-weight: 600;
    margin: 0.5rem 0 1rem;
}
.avatar-circle {
    width: 58px; height: 58px; border-radius: 14px;
    background: linear-gradient(135deg, #1D4ED8, #3B82F6);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; margin: 0 auto 1rem;
    box-shadow: 0 4px 14px rgba(37,99,235,0.28);
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_connection() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

try:
    supabase = init_connection()
except Exception as e:
    st.error(f"⚠️ Database connection failed: {e}")
    st.stop()

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# ── LOGGED IN ─────────────────────────────────────────────────────────────────
if st.session_state["logged_in"]:
    st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="auth-card" style="text-align:center;">
        <div class="avatar-circle">👤</div>
        <h2 style="color:#1E293B;margin-bottom:0.25rem;font-size:1.3rem;font-weight:700;">
            Welcome Back!</h2>
        <p style="color:#64748B;font-size:0.875rem;margin-bottom:1rem;">
            Signed in as <strong style="color:#2563EB;">
            {st.session_state.get('username','')}</strong></p>
        <div class="auth-session">✅ Session Active</div>
    </div>""", unsafe_allow_html=True)

    st.info("👈 Use the sidebar to navigate to Dashboard, Data Intake, or any other module.")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚪 Sign Out", use_container_width=True):
            supabase.auth.sign_out()
            st.session_state["logged_in"] = False
            st.session_state["username"] = ""
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ── NOT LOGGED IN ─────────────────────────────────────────────────────────────
else:
    st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div class="auth-logo">
        <div class="auth-logo-icon">📊</div>
        <h1>BuzNet</h1>
        <p>Business Intelligence System — Sign in to continue</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔑 Sign In", "📝 Create Account"])

    with tab1:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        with st.form("login_form", clear_on_submit=False):
            st.markdown("""<p style="font-size:1rem;font-weight:700;color:#1E293B;margin-bottom:1rem;">
                Welcome back 👋</p>""", unsafe_allow_html=True)
            email    = st.text_input("Email Address", placeholder="you@example.com").strip().lower()
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.form_submit_button("🔓 Sign In", use_container_width=True):
                if not email or not password:
                    st.error("⚠️ Please fill in both fields.")
                else:
                    try:
                        response = supabase.auth.sign_in_with_password(
                            {"email": email, "password": password})
                        st.session_state["logged_in"] = True
                        st.session_state["username"]  = response.user.email
                        st.success("✅ Login successful! Redirecting…")
                        st.balloons()
                        st.rerun()
                    except Exception:
                        st.error("❌ Invalid email or password.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        with st.form("register_form", clear_on_submit=False):
            st.markdown("""<p style="font-size:1rem;font-weight:700;color:#1E293B;margin-bottom:0.5rem;">
                Create your account 🚀</p>""", unsafe_allow_html=True)
            st.info("💡 Use a valid email — your data is securely linked to your account.")
            new_email = st.text_input("Email Address", placeholder="you@example.com").strip().lower()
            new_pass  = st.text_input("Password", type="password", placeholder="Min 6 characters")
            confirm   = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.form_submit_button("🚀 Create Account", use_container_width=True):
                if not new_email or not new_pass:
                    st.error("⚠️ All fields are required.")
                elif len(new_pass) < 6:
                    st.warning("⚠️ Password must be at least 6 characters.")
                elif new_pass != confirm:
                    st.error("❌ Passwords do not match.")
                else:
                    try:
                        supabase.auth.sign_up({"email": new_email, "password": new_pass})
                        st.success("✅ Account created! Switch to Sign In tab.")
                        st.balloons()
                    except Exception as e:
                        if "already registered" in str(e).lower():
                            st.error("⚠️ Email already exists. Please sign in.")
                        else:
                            st.error(f"❌ Registration failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="auth-footer">
        🔒 Encrypted &amp; Secure &nbsp;•&nbsp; ☁️ Powered by Supabase &nbsp;•&nbsp; 📊 BuzNet v4.0
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
