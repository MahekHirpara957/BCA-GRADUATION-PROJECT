import streamlit as st
import pandas as pd
from datetime import datetime
import sys, os

# Handle pathing for custom utils
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from utils import inject_theme, check_login, init_connection, page_header
except ImportError:
    # Fallbacks for local environment testing
    def inject_theme(): pass
    def check_login(): pass
    def init_connection(): return None
    def page_header(i, t, s): st.title(f"{i} {t}"); st.write(s)

st.set_page_config(page_title="Data Intake — BuzNet", page_icon="📝", layout="wide")
inject_theme()
check_login()

supabase = init_connection()

page_header("📝", "Smart Data Intake",
            "Add, upload, clean your business data — optimized for performance")

tab1, tab2, tab3 = st.tabs([
    "✍️ Manual Entry", "📁 Bulk CSV Upload", "📋 View & Edit Records"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: MANUAL ENTRY
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div style="font-size:1.25rem; font-weight:bold; margin-bottom:15px;">✍️ Add Single Record</div>', unsafe_allow_html=True)

    with st.form("entry_form", clear_on_submit=True):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**📅 Record Details**")
            date      = st.date_input("Date", value=datetime.today())
            prod_name = st.text_input("Product Name", placeholder="e.g. Premium Coffee")
            prod_qty  = st.number_input("Production Quantity", min_value=0, step=1)
        with c2:
            st.markdown("**📊 Sales & Revenue**")
            st.markdown("<br>", unsafe_allow_html=True)
            sold_qty = st.number_input("Units Sold", min_value=0, step=1)
            revenue  = st.number_input("Total Revenue (₹)", min_value=0.0, step=0.01, format="%.2f")

        st.markdown("---")
        _, col2, _ = st.columns([1, 2, 1])
        submitted = col2.form_submit_button("💾 Save Record to Cloud", width='stretch')

        if submitted:
            if not prod_name:
                st.error("❌ Product Name is required.")
            elif sold_qty > prod_qty and prod_qty > 0:
                st.error("❌ Units Sold cannot exceed Production Quantity.")
            else:
                try:
                    supabase.table("buznet_data").insert({
                        "client_id":  st.session_state.get("username", "demo_user"),
                        "Date":       date.strftime("%Y-%m-%d"),
                        "Product":    prod_name.strip(),
                        "Production": int(prod_qty),
                        "Sold":       int(sold_qty),
                        "Revenue":    float(revenue),
                    }).execute()
                    st.success(f"✅ Record for **{prod_name}** saved successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error saving: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: BULK CSV UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div style="font-size:1.25rem; font-weight:bold; margin-bottom:15px;">📁 Bulk CSV Upload</div>', unsafe_allow_html=True)
    
    # Template Download
    template_df = pd.DataFrame({
        "Date": ["2024-01-01"], "Product": ["Widget A"], 
        "Production": [100], "Sold": [80], "Revenue": [8000.00]
    })
    st.download_button("⬇️ Download CSV Template", data=template_df.to_csv(index=False).encode(),
                     file_name="buznet_template.csv", mime="text/csv")

    uploaded = st.file_uploader("📤 Upload your CSV file", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            required = ["Date", "Product", "Production", "Sold", "Revenue"]
            missing = [c for c in required if c not in df.columns]

            if missing:
                st.error(f"❌ Missing columns: {', '.join(missing)}")
            else:
                # Cleaning
                df = df.drop_duplicates()
                df['Product'] = df['Product'].fillna("Unknown")
                for c in ['Production','Sold','Revenue']:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date'])

                st.success(f"✅ Cleaned {len(df)} records.")
                st.dataframe(df.head(5), width='stretch')

                if st.button("🚀 Upload All to Cloud", width='stretch'):
                    with st.spinner("Uploading..."):
                        up_data = df.copy()
                        up_data['Date'] = up_data['Date'].dt.strftime('%Y-%m-%d')
                        up_data['client_id'] = st.session_state.get("username", "demo_user")
                        supabase.table("buznet_data").insert(up_data.to_dict(orient="records")).execute()
                        st.success("✅ Bulk upload complete!")
                        st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: VIEW, EDIT & DELETE (DATE FILTERED & NO UNIQUE ID FIX)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div style="font-size:1.25rem; font-weight:bold; margin-bottom:15px;">📋 Manage Records</div>', unsafe_allow_html=True)

    try:
        user = st.session_state.get("username", "demo_user")
        
        # 1. Select Date first to find the record
        st.markdown("### 🔍 Step 1: Filter by Date")
        filter_date = st.date_input("Select Date to view entries", value=datetime.today())
        date_str = filter_date.strftime("%Y-%m-%d")

        # Query only records for that specific date
        res = supabase.table("buznet_data").select("*").eq("client_id", user).eq("Date", date_str).execute()

        if not res.data:
            st.info(f"📭 No records found for {filter_date.strftime('%d %B %Y')}.")
        else:
            rdf = pd.DataFrame(res.data)
            
            # Show entries for that day
            st.markdown(f"### 📊 Entries for {filter_date.strftime('%d %B %Y')}")
            st.dataframe(rdf[['Product', 'Production', 'Sold', 'Revenue']], 
                         width='stretch')

            st.divider()

            # 2. Select specific entry from that day
            st.markdown("### ⚙️ Step 2: Select Entry to Action")
            action_col1, _ = st.columns([2, 1])
            
            with action_col1:
                selected_idx = st.selectbox(
                    "Select which product entry to Edit or Delete",
                    options=rdf.index,
                    format_func=lambda x: f"{rdf.iloc[x]['Product']} | ₹{rdf.iloc[x]['Revenue']:,.0f}"
                )
                selected_row = rdf.iloc[selected_idx]
                
                # Identifiers for the record (Composite Key fallback)
                orig_product = selected_row['Product']
                orig_date    = selected_row['Date']

            st.markdown("---")
            edit_col, del_col = st.columns([3, 1])

            with edit_col:
                st.markdown("**📝 Quick Edit Form**")
                with st.form("optimized_edit_form"):
                    e_c1, e_c2 = st.columns(2)
                    new_date = e_c1.date_input("Date", value=pd.to_datetime(selected_row['Date']))
                    new_prod = e_c1.text_input("Product", value=selected_row['Product'])
                    new_pq   = e_c2.number_input("Production", value=int(selected_row['Production']))
                    new_sq   = e_c2.number_input("Sold", value=int(selected_row['Sold']))
                    new_rv   = st.number_input("Revenue (₹)", value=float(selected_row['Revenue']))

                    if st.form_submit_button("💾 Update This Record", width='stretch'):
                        try:
                            # Using Composite Keys (Date + Product + Client) to update since no ID exists
                            supabase.table("buznet_data").update({
                                "Date": new_date.strftime("%Y-%m-%d"),
                                "Product": new_prod.strip(),
                                "Production": int(new_pq),
                                "Sold": int(new_sq),
                                "Revenue": float(new_rv)
                            }).eq("client_id", user).eq("Date", orig_date).eq("Product", orig_product).execute()
                            
                            st.success("✅ Record Updated!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Update failed: {e}")

            with del_col:
                st.markdown("**⚠️ Danger Zone**")
                st.write("Deleting is permanent.")
                if st.button("🗑️ Permanently Delete", type="primary", width='stretch'):
                    try:
                        # Using Composite Keys for Deletion
                        supabase.table("buznet_data").delete().eq("client_id", user).eq("Date", orig_date).eq("Product", orig_product).execute()
                        st.warning("✅ Record Deleted.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

    except Exception as e:
        st.error(f"Connection Error: {e}")

st.markdown("---")
st.caption("BuzNet Intelligence System • Data Management Module")