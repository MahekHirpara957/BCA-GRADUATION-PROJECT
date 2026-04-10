import streamlit as st
import pandas as pd
import io
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import inject_theme, check_login, init_connection, page_header

st.set_page_config(page_title="Search & Export — BuzNet", page_icon="🔍", layout="wide")
inject_theme()
check_login()

supabase = init_connection()

def load_data():
    try:
        res = supabase.table("buznet_data").select("*").eq("client_id", st.session_state["username"]).execute()
        if res.data:
            df = pd.DataFrame(res.data)
            df['Date'] = pd.to_datetime(df['Date'])
            for c in ['Production','Sold','Revenue']:
                if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error: {e}"); return pd.DataFrame()

def to_excel(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='BuzNet Data')
    return buf.getvalue()

page_header("🔍", "Search & Export Data",
            "Advanced filters • Smart search • Export to Excel & CSV • Edit & Delete records")

df = load_data()

if df.empty:
    st.markdown("""<div class="bz-card" style="text-align:center;padding:3rem;">
        <div style="font-size:4rem;">🔍</div>
        <h2>No Data Found</h2>
        <p>Add records via Data Intake first.</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── SMART SEARCH ──────────────────────────────────────────────────────────────
st.markdown('<div class="bz-section-title">🔎 Smart Search</div>', unsafe_allow_html=True)

all_products = sorted(df['Product'].unique().tolist())

sc1, sc2 = st.columns([3, 1])
with sc1:
    search_q = st.text_input("Search Products (partial match supported)",
                              placeholder="e.g. 'coffee', 'widget', 'serv…'")
with sc2:
    if search_q:
        suggestions = [p for p in all_products if search_q.lower() in p.lower()]
        if suggestions:
            st.markdown(f"""
            <div class="bz-card" style="padding:.7rem 1rem;margin-top:.5rem;">
                <div style="font-size:.75rem;color:#2563EB;font-weight:600;margin-bottom:.25rem;">💡 Suggestions</div>
                <div style="font-size:.82rem;color:#374151;">{'  •  '.join(suggestions[:5])}</div>
            </div>""", unsafe_allow_html=True)

# ── ADVANCED FILTERS ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">🎛️ Advanced Filters</div>', unsafe_allow_html=True)

with st.expander("Show / Hide Filters", expanded=True):
    fc1, fc2, fc3 = st.columns(3)

    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    date_range = fc1.date_input("📅 Date Range", value=(min_date, max_date))

    sel_prods = fc2.multiselect("📦 Products", all_products,
                                default=all_products[:5] if len(all_products) > 5 else all_products)

    sort_col = fc3.selectbox("🔃 Sort By", ["Date","Revenue","Sold","Production"])

    min_rev = float(df['Revenue'].min())
    max_rev = float(df['Revenue'].max())
    if min_rev == max_rev: max_rev = min_rev + 1
    rev_range = st.slider("💰 Revenue Range (₹)", min_rev, max_rev,
                          (min_rev, max_rev), format="₹%.0f")

# ── APPLY FILTERS ─────────────────────────────────────────────────────────────
fdf = df.copy()

if isinstance(date_range, tuple) and len(date_range) == 2:
    fdf = fdf[(fdf['Date'].dt.date >= date_range[0]) & (fdf['Date'].dt.date <= date_range[1])]

if sel_prods:
    fdf = fdf[fdf['Product'].isin(sel_prods)]

fdf = fdf[(fdf['Revenue'] >= rev_range[0]) & (fdf['Revenue'] <= rev_range[1])]

if search_q:
    fdf = fdf[fdf['Product'].str.contains(search_q, case=False, na=False)]

fdf = fdf.sort_values(sort_col, ascending=False)

# ── SUMMARY METRICS ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">📊 Results Summary</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Records Found",   len(fdf))
m2.metric("Total Revenue",   f"₹{fdf['Revenue'].sum():,.0f}")
m3.metric("Units Sold",      f"{fdf['Sold'].sum():,.0f}")
m4.metric("Unique Products", fdf['Product'].nunique())

# ── EXPORT ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">📥 Export Filtered Data</div>', unsafe_allow_html=True)

export_df = fdf.drop(columns=['client_id','id'], errors='ignore').copy()
export_df['Date'] = export_df['Date'].dt.strftime('%Y-%m-%d')

ec1, ec2, ec3 = st.columns([1, 1, 2])
with ec1:
    st.download_button(
        "⬇️ Download Excel (.xlsx)",
        data=to_excel(export_df),
        file_name="buznet_filtered_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True)
with ec2:
    st.download_button(
        "⬇️ Download CSV",
        data=export_df.to_csv(index=False).encode(),
        file_name="buznet_filtered_data.csv",
        mime="text/csv",
        use_container_width=True)
with ec3:
    st.markdown(f"""
    <div class="bz-card" style="padding:.9rem 1.2rem;">
        <span style="color:#6B7280;font-size:.85rem;">
            💡 Exporting <strong>{len(fdf)} records</strong> across
            <strong>{fdf['Product'].nunique()} products</strong> — ready for business reporting
        </span>
    </div>""", unsafe_allow_html=True)

# ── DATA TABLE ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">📋 Filtered Records</div>', unsafe_allow_html=True)

if fdf.empty:
    st.markdown("""<div class="bz-card" style="text-align:center;padding:2rem;">
        <div style="font-size:2.5rem;">🔍</div>
        <p style="color:#6B7280;">No records match your filters. Try adjusting them.</p>
    </div>""", unsafe_allow_html=True)
else:
    # View mode toggle
    view_mode = st.radio("View Mode:", ["📋 Table View", "✏️ Edit & Delete Records"], horizontal=True)

    if view_mode == "📋 Table View":
        st.dataframe(export_df, hide_index=True, use_container_width=True,
            column_config={
                "Date":       st.column_config.TextColumn("📅 Date"),
                "Product":    st.column_config.TextColumn("📦 Product"),
                "Production": st.column_config.NumberColumn("🏭 Produced",  format="%d"),
                "Sold":       st.column_config.NumberColumn("📊 Sold",      format="%d"),
                "Revenue":    st.column_config.NumberColumn("💰 Revenue (₹)", format="₹%.2f"),
            })

    else:
        st.info("💡 Expand any record to edit its values or delete it.")

        for idx, row in fdf.iterrows():
            rec_id = row.get('id', idx)
            with st.expander(
                f"📦 {row['Product']}  |  📅 {row['Date'].strftime('%d %b %Y')}  |  "
                f"💰 ₹{float(row['Revenue']):,.2f}  |  📊 Sold: {int(row['Sold'])}"
            ):
                col_e, col_d = st.columns([4, 1])

                with col_e:
                    with st.form(f"search_edit_{rec_id}", clear_on_submit=False):
                        ec1, ec2 = st.columns(2)
                        new_date     = ec1.date_input("Date", value=row['Date'].date(), key=f"sd_{rec_id}")
                        new_prod     = ec1.text_input("Product", value=row['Product'], key=f"sp_{rec_id}")
                        new_prod_qty = ec2.number_input("Production", value=int(row['Production']),
                                                         min_value=0, key=f"spq_{rec_id}")
                        new_sold     = ec2.number_input("Sold", value=int(row['Sold']),
                                                         min_value=0, key=f"ss_{rec_id}")
                        new_rev      = ec2.number_input("Revenue (₹)", value=float(row['Revenue']),
                                                         min_value=0.0, format="%.2f", key=f"sr_{rec_id}")
                        if st.form_submit_button("💾 Save Changes", use_container_width=True):
                            try:
                                supabase.table("buznet_data").update({
                                    "Date":       new_date.strftime("%Y-%m-%d"),
                                    "Product":    new_prod.strip(),
                                    "Production": int(new_prod_qty),
                                    "Sold":       int(new_sold),
                                    "Revenue":    float(new_rev),
                                }).eq("id", rec_id).execute()
                                st.success("✅ Record updated!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Update failed: {e}")

                with col_d:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if st.button("🗑️ Delete", key=f"sdel_{rec_id}", use_container_width=True):
                        if st.session_state.get(f"sconfirm_{rec_id}"):
                            try:
                                supabase.table("buznet_data").delete().eq("id", rec_id).execute()
                                st.success("✅ Deleted!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Delete failed: {e}")
                        else:
                            st.session_state[f"sconfirm_{rec_id}"] = True
                            st.warning("⚠️ Click Delete again to confirm.")

    if len(fdf) > 0:
        st.markdown("---")
        ic1, ic2 = st.columns(2)
        top_p = fdf.groupby('Product')['Revenue'].sum().idxmax()
        top_r = fdf.groupby('Product')['Revenue'].sum().max()
        ic1.success(f"🏆 **Top Product in Results:** {top_p} — ₹{top_r:,.0f}")
        ic2.info(f"📈 **Avg Revenue per Record:** ₹{fdf['Revenue'].mean():,.2f}")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="bz-card" style="border-left:4px solid #2563EB;">
    <strong>💡 Search & Export Tips</strong>
    <ul style="margin:.5rem 0 0 1rem;color:#374151;font-size:.88rem;">
        <li>Partial text search — "cof" matches "Coffee Premium"</li>
        <li>Switch to <strong>Edit & Delete Records</strong> view to update or remove individual entries</li>
        <li>Combine date range + product filter for precise reporting</li>
        <li>Excel export is ready for direct presentation use</li>
    </ul>
</div>""", unsafe_allow_html=True)
