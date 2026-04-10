import streamlit as st
import pandas as pd
from datetime import datetime
from supabase import create_client, Client
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import inject_theme, check_login, init_connection, page_header

st.set_page_config(page_title="Data Intake — BuzNet", page_icon="📝", layout="wide")
inject_theme()
check_login()

supabase = init_connection()

page_header("📝", "Smart Data Intake",
            "Add, upload, clean your business data — edit & delete records inline")

tab1, tab2, tab3  = st.tabs([
    "✍️ Manual Entry", "📁 Bulk CSV Upload", "📋 View & Edit Records"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: MANUAL ENTRY
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="bz-section-title">✍️ Add Single Record</div>', unsafe_allow_html=True)

    with st.form("entry_form", clear_on_submit=True):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**📅 Record Details**")
            date      = st.date_input("Date", value=datetime.today())
            prod_name = st.text_input("Product Name", placeholder="e.g. Premium Coffee, Widget A")
            prod_qty  = st.number_input("Production Quantity", min_value=0, step=1)
        with c2:
            st.markdown("**📊 Sales & Revenue**")
            st.markdown("<br>", unsafe_allow_html=True)
            sold_qty = st.number_input("Units Sold", min_value=0, step=1)
            revenue  = st.number_input("Total Revenue (₹)", min_value=0.0, step=0.01, format="%.2f")

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("💾 Save Record to Cloud", use_container_width=True)

        if submitted:
            errors = []
            if not prod_name:
                errors.append("Product Name is required.")
            if sold_qty > prod_qty and prod_qty > 0:
                errors.append("Units Sold cannot exceed Production Quantity.")
            if errors:
                for e in errors: st.error(f"❌ {e}")
            else:
                try:
                    supabase.table("buznet_data").insert({
                        "client_id":  st.session_state["username"],
                        "Date":       date.strftime("%Y-%m-%d"),
                        "Product":    prod_name.strip(),
                        "Production": int(prod_qty),
                        "Sold":       int(sold_qty),
                        "Revenue":    float(revenue),
                    }).execute()
                    st.success(f"✅ Record for **{prod_name}** saved on {date.strftime('%B %d, %Y')}!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error saving: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: BULK CSV UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="bz-section-title">📁 Bulk CSV Upload</div>', unsafe_allow_html=True)
    st.info("💡 Upload a CSV with multiple records. Download the template to ensure correct format.")

    col_t1, col_t2, col_t3 = st.columns([1, 2, 1])
    with col_t2:
        template_df = pd.DataFrame({
            "Date":       ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Product":    ["Coffee Premium", "Widget A", "Service B"],
            "Production": [100, 150, 200],
            "Sold":       [80, 130, 180],
            "Revenue":    [8000.00, 13000.00, 18000.00],
        })
        st.download_button("⬇️ Download CSV Template",
            data=template_df.to_csv(index=False).encode(),
            file_name="buznet_template.csv", mime="text/csv",
            use_container_width=True)

    st.divider()
    uploaded = st.file_uploader("📤 Upload your CSV file", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.markdown("### 👁️ Preview (first 10 rows)")
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)

            required = ["Date", "Product", "Production", "Sold", "Revenue"]
            missing  = [c for c in required if c not in df.columns]

            if missing:
                st.error(f"❌ Missing columns: {', '.join(missing)}")
            else:
                st.markdown("### 🧹 Auto Data Cleaning")
                orig_len = len(df)
                df = df.drop_duplicates()
                dupes = orig_len - len(df)
                nulls_before = df.isnull().sum().sum()
                df['Product'] = df['Product'].fillna("Unknown")
                for c in ['Production','Sold','Revenue']:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                bad_dates = df['Date'].isna().sum()
                df = df.dropna(subset=['Date'])

                cl1, cl2, cl3, cl4 = st.columns(4)
                cl1.metric("Original Rows", orig_len)
                cl2.metric("Duplicates Removed", dupes, delta=f"-{dupes}" if dupes else "✅ None")
                cl3.metric("Nulls Fixed", int(nulls_before - df.isnull().sum().sum()))
                cl4.metric("Bad Dates Dropped", bad_dates)
                st.success(f"✅ Clean data: **{len(df)} records** ready to upload")

                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                sc1.metric("Records", len(df))
                sc2.metric("Products", df['Product'].nunique())
                sc3.metric("Total Revenue", f"₹{df['Revenue'].sum():,.0f}")
                sc4.metric("Avg Revenue", f"₹{df['Revenue'].mean():,.0f}")
                sc5.metric("Date Range", f"{df['Date'].min().strftime('%d %b')} – {df['Date'].max().strftime('%d %b %y')}")

                st.markdown("<br>", unsafe_allow_html=True)
                col_up1, col_up2, col_up3 = st.columns([1, 2, 1])
                with col_up2:
                    if st.button("🚀 Upload Clean Data to Cloud", use_container_width=True):
                        with st.spinner("Uploading…"):
                            try:
                                up = df[required].copy()
                                up['Date']       = up['Date'].dt.strftime('%Y-%m-%d')
                                up['Production'] = up['Production'].astype(int)
                                up['Sold']       = up['Sold'].astype(int)
                                up['Revenue']    = up['Revenue'].astype(float)
                                up['client_id']  = st.session_state["username"]
                                supabase.table("buznet_data").insert(up.to_dict(orient="records")).execute()
                                st.success(f"✅ Uploaded **{len(up)} records** successfully!")
                                st.balloons()
                            except Exception as e:
                                st.error(f"❌ Upload failed: {e}")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: VIEW, EDIT & DELETE RECORDS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="bz-section-title">📋 Your Records — Edit & Delete</div>', unsafe_allow_html=True)

    try:
        res = supabase.table("buznet_data").select("*").eq(
            "client_id", st.session_state["username"]
        ).order("Date", desc=True).execute()

        if not res.data:
            st.info("📭 No records found. Add records from the Manual Entry tab.")
        else:
            rdf = pd.DataFrame(res.data)
            rdf['Date'] = pd.to_datetime(rdf['Date'])

            # Summary
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Records", len(rdf))
            m2.metric("Total Revenue", f"₹{rdf['Revenue'].sum():,.0f}")
            m3.metric("Unique Products", rdf['Product'].nunique())

            st.markdown("---")

            # Filter
            products_list = ["All"] + sorted(rdf['Product'].unique().tolist())
            sel_prod = st.selectbox("🔍 Filter by Product", products_list)
            view_df = rdf if sel_prod == "All" else rdf[rdf['Product'] == sel_prod]

            st.markdown(f"Showing **{len(view_df)}** records")
            st.markdown("<br>", unsafe_allow_html=True)

            # Display records with edit/delete
            for idx, row in view_df.iterrows():
                rec_id = row.get('id', idx)
                with st.expander(
                    f"📦 {row['Product']}  |  📅 {row['Date'].strftime('%d %b %Y')}  |  "
                    f"💰 ₹{float(row['Revenue']):,.2f}  |  📊 Sold: {int(row['Sold'])}"
                ):
                    col_e, col_d = st.columns([4, 1])

                    with col_e:
                        with st.form(f"edit_form_{rec_id}", clear_on_submit=False):
                            ec1, ec2 = st.columns(2)
                            new_date  = ec1.date_input("Date", value=row['Date'].date(), key=f"d_{rec_id}")
                            new_prod  = ec1.text_input("Product", value=row['Product'], key=f"p_{rec_id}")
                            new_prod_qty = ec2.number_input("Production", value=int(row['Production']),
                                                             min_value=0, key=f"pq_{rec_id}")
                            new_sold  = ec2.number_input("Sold", value=int(row['Sold']),
                                                          min_value=0, key=f"s_{rec_id}")
                            new_rev   = ec2.number_input("Revenue (₹)", value=float(row['Revenue']),
                                                          min_value=0.0, format="%.2f", key=f"r_{rec_id}")

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
                        if st.button("🗑️ Delete", key=f"del_{rec_id}", use_container_width=True):
                            if st.session_state.get(f"confirm_del_{rec_id}"):
                                try:
                                    supabase.table("buznet_data").delete().eq("id", rec_id).execute()
                                    st.success("✅ Deleted!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Delete failed: {e}")
                            else:
                                st.session_state[f"confirm_del_{rec_id}"] = True
                                st.warning("⚠️ Click Delete again to confirm.")

    except Exception as e:
        st.error(f"Error loading records: {e}")



st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="bz-card" style="border-left:4px solid #2563EB;">
    <strong>💡 Pro Tips</strong>
    <ul style="margin:.5rem 0 0 1rem;color:#374151;font-size:.88rem;">
        <li>Keep product names consistent (case-sensitive matching)</li>
        <li>Date format must be <code>YYYY-MM-DD</code> for CSV uploads</li>
        <li>Units Sold should not exceed Production Quantity</li>
        <li>Use the "View & Edit Records" tab to update or delete individual entries</li>
    </ul>
</div>""", unsafe_allow_html=True)
