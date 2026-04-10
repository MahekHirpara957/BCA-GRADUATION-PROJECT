import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import inject_theme, check_login, init_connection, page_header, format_currency

st.set_page_config(page_title="Settings — BuzNet", page_icon="⚙️", layout="centered")
inject_theme()
check_login()

supabase = init_connection()

def load_user_data():
    try:
        res = supabase.table("buznet_data").select("*").eq("client_id", st.session_state["username"]).execute()
        if res.data:
            df = pd.DataFrame(res.data)
            df['Date'] = pd.to_datetime(df['Date'])
            for c in ['Production','Sold','Revenue']:
                if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            return df
        return pd.DataFrame()
    except: return pd.DataFrame()

page_header("⚙️", "Settings & Preferences", "Customise your BuzNet experience")

# ── ACCOUNT ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">👤 Account Information</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="bz-card">
    <div style="display:flex;align-items:center;gap:1.2rem;">
        <div style="width:52px;height:52px;border-radius:14px;
            background:linear-gradient(135deg,#1D4ED8,#3B82F6);
            display:flex;align-items:center;justify-content:center;
            font-size:1.4rem;color:white;
            box-shadow:0 4px 12px rgba(37,99,235,0.25);">👤</div>
        <div>
            <div style="font-weight:700;color:#1E293B;font-size:1rem;">
                {st.session_state.get('username','—')}</div>
            <div style="color:#64748B;font-size:0.82rem;margin-top:0.1rem;">
                Logged In &nbsp;•&nbsp; Active User</div>
        </div>
        <div style="margin-left:auto;">
            <span style="background:#ECFDF5;color:#10B981;
                padding:0.3rem 0.9rem;border-radius:999px;
                font-weight:700;font-size:0.78rem;
                border:1px solid #D1FAE5;">✅ Authenticated</span>
        </div>
    </div>
</div>""", unsafe_allow_html=True)

# ── CURRENCY ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">💱 Currency Settings</div>', unsafe_allow_html=True)

currencies = {"Indian Rupee ₹":"₹","US Dollar $":"$","Euro €":"€",
              "British Pound £":"£","Japanese Yen ¥":"¥","UAE Dirham":"د.إ"}
cur_sym   = st.session_state.get("currency_symbol","₹")
cur_label = next((k for k,v in currencies.items() if v==cur_sym),"Indian Rupee ₹")
sel_cur   = st.selectbox("💱 Currency Symbol",
                         list(currencies.keys()),
                         index=list(currencies.keys()).index(cur_label))
if st.button("💾 Save Currency"):
    st.session_state["currency_symbol"] = currencies[sel_cur]
    st.success(f"✅ Currency updated to {currencies[sel_cur]}")
    st.rerun()
st.caption(f"Preview: {format_currency(12345.67)}")

# ── NOTIFICATIONS ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">🔔 Notification Preferences</div>', unsafe_allow_html=True)

if "notif" not in st.session_state:
    st.session_state["notif"] = {"dashboard_alerts":True,"forecast_alerts":True,
                                  "data_quality":True,"weekly_report":False}
nc1, nc2 = st.columns(2)
with nc1:
    n1 = st.toggle("📊 Dashboard Alerts",     value=st.session_state["notif"]["dashboard_alerts"])
    n2 = st.toggle("🔮 Forecast Alerts",       value=st.session_state["notif"]["forecast_alerts"])
with nc2:
    n3 = st.toggle("🧹 Data Quality Warnings", value=st.session_state["notif"]["data_quality"])
    n4 = st.toggle("📧 Weekly Summary Report", value=st.session_state["notif"]["weekly_report"])

if st.button("💾 Save Notification Settings"):
    st.session_state["notif"] = {"dashboard_alerts":n1,"forecast_alerts":n2,
                                  "data_quality":n3,"weekly_report":n4}
    st.success("✅ Notification preferences saved!")

# ── CHANGE PASSWORD ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">🔑 Change Password</div>', unsafe_allow_html=True)

with st.form("change_pw"):
    new_pw  = st.text_input("New Password", type="password", placeholder="Min 6 characters")
    conf_pw = st.text_input("Confirm Password", type="password", placeholder="Re-enter")
    if st.form_submit_button("🔑 Update Password", use_container_width=True):
        if len(new_pw) < 6:
            st.error("❌ Password must be ≥ 6 characters.")
        elif new_pw != conf_pw:
            st.error("❌ Passwords do not match.")
        else:
            try:
                supabase.auth.update_user({"password": new_pw})
                st.success("✅ Password updated successfully!")
            except Exception as e:
                st.error(f"❌ Failed: {e}")

# ── DATA SUMMARY ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">📊 Your Data Summary</div>', unsafe_allow_html=True)
df = load_user_data()
if not df.empty:
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Records", len(df))
    d2.metric("Total Revenue", format_currency(df['Revenue'].sum()))
    d3.metric("Products",      df['Product'].nunique() if 'Product' in df.columns else 0)
    d4.metric("Date Range",    f"{df['Date'].min().strftime('%d %b')} – {df['Date'].max().strftime('%d %b %y')}")
else:
    st.info("📭 No data yet.")

# ── EXPORT ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">📤 Export Your Data</div>', unsafe_allow_html=True)
if not df.empty:
    exp = df.drop(columns=['client_id'], errors='ignore').copy()
    exp['Date'] = exp['Date'].dt.strftime('%Y-%m-%d')
    st.download_button("⬇️ Download All Data (CSV)",
        data=exp.to_csv(index=False).encode(),
        file_name="buznet_all_data.csv", mime="text/csv",
        use_container_width=True)
else:
    st.info("📭 No data to export.")

# ── DANGER ZONE ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="background:#FFF1F2;border:1.5px solid #FECDD3;
     border-radius:14px;padding:1.4rem;">
    <div style="font-size:1rem;font-weight:700;color:#E11D48;margin-bottom:0.3rem;">
        🚨 Danger Zone</div>
    <p style="color:#64748B;font-size:0.875rem;margin:0;">
        Permanent, irreversible actions — proceed with caution.</p>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("⚠️ I understand the risks — Show delete options"):
    st.warning("This will **permanently delete ALL your business records**. Export first as backup.")
    confirm_txt = st.text_input('Type "DELETE" in all caps to confirm:', placeholder="DELETE", key="del_confirm")
    if st.button("🗑️ Delete All My Data", use_container_width=True):
        if confirm_txt.strip() == "DELETE":
            try:
                supabase.table("buznet_data").delete().eq("client_id", st.session_state["username"]).execute()
                st.success("✅ All data permanently deleted.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Delete failed: {e}")
        else:
            st.warning('Type "DELETE" exactly.')

# ── ABOUT ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="bz-card">
    <div class="bz-section-title">ℹ️ About BuzNet</div>
    <table style="width:100%;border-collapse:collapse;font-size:.875rem;">
        <tr><td style="padding:.5rem;color:#64748B;font-weight:600;border-bottom:1px solid #E2E8F0;">Application</td>
            <td style="padding:.5rem;color:#1E293B;border-bottom:1px solid #E2E8F0;">BuzNet v4.0 — Business Intelligence System</td></tr>
        <tr><td style="padding:.5rem;color:#64748B;font-weight:600;border-bottom:1px solid #E2E8F0;">Project Type</td>
            <td style="padding:.5rem;color:#1E293B;border-bottom:1px solid #E2E8F0;">BCA Major Final Year Project — 20 Credits / 1200 Marks</td></tr>
        <tr><td style="padding:.5rem;color:#64748B;font-weight:600;border-bottom:1px solid #E2E8F0;">Framework</td>
            <td style="padding:.5rem;color:#1E293B;border-bottom:1px solid #E2E8F0;">Python 3.11 + Streamlit</td></tr>
        <tr><td style="padding:.5rem;color:#64748B;font-weight:600;border-bottom:1px solid #E2E8F0;">Database</td>
            <td style="padding:.5rem;color:#1E293B;border-bottom:1px solid #E2E8F0;">Supabase (PostgreSQL)</td></tr>
        <tr><td style="padding:.5rem;color:#64748B;font-weight:600;border-bottom:1px solid #E2E8F0;">ML Engine</td>
            <td style="padding:.5rem;color:#1E293B;border-bottom:1px solid #E2E8F0;">Scikit-learn + XGBoost (LR, RF, DT, XGB)</td></tr>
        <tr><td style="padding:.5rem;color:#64748B;font-weight:600;">Institution</td>
            <td style="padding:.5rem;color:#2563EB;font-weight:700;">CMPICA, CHARUSAT — Changa</td></tr>
    </table>
</div>""", unsafe_allow_html=True)
