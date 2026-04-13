import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import sys, os, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import inject_theme, check_login, init_connection, page_header

st.set_page_config(page_title="Auto Insights — BuzNet", page_icon="💡", layout="wide")
inject_theme()
check_login()

supabase = init_connection()
BLUE = "#2563EB"; LBLUE = "#3B82F6"; GREEN = "#10B981"; INDIGO = "#6366F1"; AMBER = "#F59E0B"

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

def generate_pdf(df, insights):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib.colors import HexColor, white
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER

        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
            topMargin=1.5*cm, bottomMargin=1.5*cm,
            leftMargin=2*cm,  rightMargin=2*cm)

        styles  = getSampleStyleSheet()
        purple  = HexColor("#2563EB")
        pink    = HexColor("#3B82F6")
        dark    = HexColor("#1F2937")
        gray    = HexColor("#6B7280")

        title_s   = ParagraphStyle('T', fontSize=26, textColor=purple, spaceAfter=4,
                                    alignment=TA_CENTER, fontName='Helvetica-Bold')
        sub_s     = ParagraphStyle('S', fontSize=11, textColor=gray,
                                    alignment=TA_CENTER, spaceAfter=16)
        h2_s      = ParagraphStyle('H2', fontSize=13, textColor=purple,
                                    spaceBefore=14, spaceAfter=6, fontName='Helvetica-Bold')
        body_s    = ParagraphStyle('B', fontSize=10, textColor=dark, spaceAfter=4, leading=15)
        insight_s = ParagraphStyle('I', fontSize=10, textColor=dark,
                                    leftIndent=16, spaceAfter=5, leading=15)

        story = []
        story.append(Paragraph("BuzNet", title_s))
        story.append(Paragraph("Business Intelligence System — Auto-Generated Report", sub_s))
        story.append(Paragraph(
            f"Generated: {datetime.datetime.now().strftime('%d %B %Y, %I:%M %p')}", sub_s))
        story.append(HRFlowable(width="100%", thickness=2, color=purple, spaceAfter=14))

        story.append(Paragraph("Key Performance Indicators", h2_s))
        total_rev  = df['Revenue'].sum()
        total_sold = df['Sold'].sum()
        total_prod = df['Production'].sum()
        avg_price  = total_rev / total_sold if total_sold > 0 else 0

        kpi_data = [
            ['Metric', 'Value'],
            ['Total Revenue',       f"Rs {total_rev:,.2f}"],
            ['Total Units Sold',    f"{total_sold:,.0f}"],
            ['Total Production',    f"{total_prod:,.0f}"],
            ['Avg Unit Price',      f"Rs {avg_price:,.2f}"],
            ['Unique Products',     str(df['Product'].nunique())],
            ['Total Records',       str(len(df))],
            ['Date Range',          f"{df['Date'].min().strftime('%d %b %Y')} - {df['Date'].max().strftime('%d %b %Y')}"],
        ]
        t = Table(kpi_data, colWidths=[7*cm, 9*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND',   (0,0), (-1,0), purple),
            ('TEXTCOLOR',    (0,0), (-1,0), white),
            ('FONTNAME',     (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',     (0,0), (-1,0), 11),
            ('ROWBACKGROUNDS',(0,1),(-1,-1), [HexColor("#F8FAFC"), white]),
            ('FONTSIZE',     (0,1), (-1,-1), 10),
            ('GRID',         (0,0), (-1,-1), 0.5, HexColor("#DBEAFE")),
            ('TOPPADDING',   (0,0), (-1,-1), 6),
            ('BOTTOMPADDING',(0,0), (-1,-1), 6),
            ('LEFTPADDING',  (0,0), (-1,-1), 12),
        ]))
        story.append(t)
        story.append(Spacer(1, 14))

        story.append(Paragraph("Auto-Generated Business Insights", h2_s))
        for ins in insights:
            clean = (ins.replace("**","").replace("🟢","Good:").replace("🔴","Alert:")
                    .replace("⚠️","Warning:").replace("📈","Up:").replace("📉","Down:"))
            story.append(Paragraph(f"• {clean}", insight_s))
        story.append(Spacer(1, 14))

        story.append(Paragraph("Product Performance Summary", h2_s))
        prod_sum = df.groupby('Product').agg(
            Revenue=('Revenue','sum'), Sold=('Sold','sum'),
            Production=('Production','sum')).reset_index().sort_values('Revenue', ascending=False)
        prod_data = [['Product','Revenue (Rs)','Units Sold','Production']]
        for _, row in prod_sum.iterrows():
            prod_data.append([str(row['Product']), f"{row['Revenue']:,.0f}",
                              f"{row['Sold']:,.0f}", f"{row['Production']:,.0f}"])
        pt = Table(prod_data, colWidths=[6*cm,4*cm,4*cm,4*cm])
        pt.setStyle(TableStyle([
            ('BACKGROUND',   (0,0), (-1,0), pink),
            ('TEXTCOLOR',    (0,0), (-1,0), white),
            ('FONTNAME',     (0,0), (-1,0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor("#F8FAFC"), white]),
            ('FONTSIZE',     (0,0), (-1,-1), 9),
            ('GRID',         (0,0), (-1,-1), 0.5, HexColor("#DBEAFE")),
            ('TOPPADDING',   (0,0), (-1,-1), 5),
            ('BOTTOMPADDING',(0,0), (-1,-1), 5),
            ('LEFTPADDING',  (0,0), (-1,-1), 8),
        ]))
        story.append(pt)
        story.append(Spacer(1, 20))

        story.append(HRFlowable(width="100%", thickness=1, color=purple, spaceAfter=8))
        story.append(Paragraph("BuzNet v3.0 — B.Tech Major Project | CMPICA, CHARUSAT", sub_s))

        doc.build(story)
        return buf.getvalue()
    except ImportError:
        return None

page_header("💡", "Auto Insights & Reports",
            "AI-powered business insights + downloadable professional PDF reports")

df = load_data()
if df.empty:
    st.markdown("""<div class="bz-card" style="text-align:center;padding:3rem;">
        <div style="font-size:4rem;">💡</div>
        <h2>No Data Found</h2><p>Add records via Data Intake first.</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── AUTO INSIGHTS ENGINE ──────────────────────────────────────────────────────
st.markdown('<div class="bz-section-title">🤖 Auto-Generated Business Insights</div>', unsafe_allow_html=True)

now_m   = df['Date'].max().to_period('M')
prev_m  = now_m - 1
this_df = df[df['Date'].dt.to_period('M') == now_m]
prev_df = df[df['Date'].dt.to_period('M') == prev_m]

this_rev  = this_df['Revenue'].sum()
prev_rev  = prev_df['Revenue'].sum()
this_sold = this_df['Sold'].sum()
prev_sold = prev_df['Sold'].sum()

rev_change  = ((this_rev - prev_rev) / prev_rev * 100) if prev_rev > 0 else 0
sold_change = ((this_sold - prev_sold) / prev_sold * 100) if prev_sold > 0 else 0

total_rev   = df['Revenue'].sum()
total_sold  = df['Sold'].sum()
total_prod  = df['Production'].sum()
sold_rate   = (total_sold / total_prod * 100) if total_prod > 0 else 0
avg_price   = total_rev / total_sold if total_sold > 0 else 0

best_product  = df.groupby('Product')['Revenue'].sum().idxmax()
worst_product = df.groupby('Product')['Revenue'].sum().idxmin()

insights = []

if rev_change >= 20:
    insights.append(f"🟢 **Revenue surged {rev_change:.1f}% this month** — ₹{this_rev:,.0f} vs ₹{prev_rev:,.0f} last month. Excellent growth!")
elif rev_change >= 5:
    insights.append(f"📈 **Revenue increased {rev_change:.1f}% this month** — Positive momentum continuing.")
elif rev_change < -10:
    insights.append(f"🔴 **Revenue declined {abs(rev_change):.1f}% this month** — Immediate review recommended.")
elif prev_rev == 0:
    insights.append(f"📈 **First month of data** — Revenue recorded: ₹{this_rev:,.0f}")
else:
    insights.append(f"⚠️ **Revenue was flat this month** (change: {rev_change:+.1f}%) — Focus on growth strategies.")

if sold_change >= 15:
    insights.append(f"🟢 **Sales units grew {sold_change:.1f}% this month** — Demand is accelerating.")
elif sold_change < -15 and prev_sold > 0:
    insights.append(f"🔴 **Sales units dropped {abs(sold_change):.1f}%** — Investigate demand drivers.")

if sold_rate >= 90:
    insights.append(f"🟢 **Outstanding sell-through rate: {sold_rate:.1f}%** — Demand exceeds production. Scale up!")
elif sold_rate >= 70:
    insights.append(f"📈 **Good sell-through rate: {sold_rate:.1f}%** — Production and demand are well balanced.")
elif sold_rate < 50:
    insights.append(f"⚠️ **Low sell-through rate: {sold_rate:.1f}%** — Overproduction detected. Optimise planning.")

insights.append(f"🏆 **Best performing product: {best_product}** — Focus marketing here for maximum ROI.")
if best_product != worst_product:
    insights.append(f"📉 **Underperforming product: {worst_product}** — Review pricing, quality, or demand.")

days_old = (pd.Timestamp.now() - df['Date'].max()).days
if days_old > 30:
    insights.append(f"⏰ **Data is {days_old} days old** — Add recent entries for accurate analysis.")

insights.append(f"💵 **Average unit price: ₹{avg_price:,.2f}** across all products.")

for ins in insights:
    color  = "#F0FDF4" if "🟢" in ins else "#FEF2F2" if "🔴" in ins else "#FFFBEB" if "⚠️" in ins else "#EFF6FF"
    border = GREEN if "🟢" in ins else "#F43F5E" if "🔴" in ins else "#F59E0B" if "⚠️" in ins else "#2563EB"
    st.markdown(f"""
    <div style="background:{color};border-left:4px solid {border};border-radius:10px;
                padding:.9rem 1.1rem;margin-bottom:.6rem;">{ins}</div>""",
                unsafe_allow_html=True)

# ── KPI SNAPSHOT ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">📊 Insight Metrics Snapshot</div>', unsafe_allow_html=True)

ic1, ic2, ic3, ic4 = st.columns(4)
ic1.metric("This Month Revenue",    f"₹{this_rev:,.0f}",      f"{rev_change:+.1f}% vs last month")
ic2.metric("This Month Sales",      f"{this_sold:,.0f} units", f"{sold_change:+.1f}% vs last month")
ic3.metric("Overall Sell-Through",  f"{sold_rate:.1f}%")
ic4.metric("Avg Unit Price",        f"₹{avg_price:,.2f}")

# ── TIMELINE CHART ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">📈 Monthly Performance Timeline</div>', unsafe_allow_html=True)

monthly = df.set_index('Date').resample('ME')[['Revenue','Sold','Production']].sum().reset_index()
fig = go.Figure()
fig.add_trace(go.Bar(x=monthly['Date'], y=monthly['Revenue'],
    name='Revenue', marker_color=BLUE, opacity=0.85))
fig.add_trace(go.Scatter(x=monthly['Date'], y=monthly['Sold'],
    mode='lines+markers', name='Units Sold',
    line=dict(color=LBLUE, width=3), yaxis='y2'))
fig.update_layout(
    title="Monthly Revenue & Sales Timeline",
    plot_bgcolor='white', paper_bgcolor='white',
    hovermode='x unified', font=dict(family='Inter'),
    yaxis=dict(title='Revenue (₹)'),
    yaxis2=dict(title='Units Sold', overlaying='y', side='right'),
    legend=dict(orientation='h', y=-0.15))
st.plotly_chart(fig, width='stretch')

# ── PDF REPORT ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">📄 Download PDF Business Report</div>', unsafe_allow_html=True)

st.markdown("""
<div class="bz-card" style="border-left:4px solid #2563EB;margin-bottom:1rem;">
    <div style="font-weight:700;color:#1F2937;margin-bottom:.3rem;">🗂️ Report Contents</div>
    <div style="color:#6B7280;font-size:.88rem;line-height:1.8;">
        ✅ Executive KPI Summary  •  ✅ All Auto-Generated Insights  •
        ✅ Product Performance Table  •  ✅ Date Range Overview  •
        ✅ Professional BuzNet Branding
    </div>
</div>""", unsafe_allow_html=True)

pc1, pc2, pc3 = st.columns([1, 2, 1])
with pc2:
    if st.button("⚡ Generate PDF Report", width='stretch'):
        st.session_state['gen_report'] = True

    if st.session_state.get('gen_report', False):
        with st.spinner("📄 Generating PDF…"):
            plain = [ins.replace("**","") for ins in insights]
            pdf_bytes = generate_pdf(df, plain)

        if pdf_bytes:
            st.download_button(
                "⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=f"buznet_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                width='stretch')
            st.success("✅ PDF ready — click above to download!")
        else:
            st.error("❌ Install reportlab: `pip install reportlab`")

# ── GEMINI AI (Optional) ──────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🤖 Ask Gemini AI for Advanced Insights (Optional)"):
    st.info("Add `GOOGLE_API_KEY` to `.streamlit/secrets.toml` to enable this feature.")
    try:
        import google.generativeai as genai
        GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", None)
        if GOOGLE_API_KEY:
            ai_q = st.text_input("Ask about your business data:",
                                  placeholder="e.g. What are my top growth opportunities?")
            if st.button("Ask Gemini") and ai_q:
                genai.configure(api_key=GOOGLE_API_KEY)
                model = genai.GenerativeModel('gemini-pro')
                context = f"""
                Business Data Summary:
                - Total Revenue: Rs {total_rev:,.0f}
                - Units Sold: {total_sold:,.0f}
                - Avg Unit Price: Rs {avg_price:,.2f}
                - Products: {df['Product'].nunique()}
                - Monthly Revenue Change: {rev_change:+.1f}%
                - Sell-Through Rate: {sold_rate:.1f}%

                User Question: {ai_q}
                Provide actionable business advice.
                """
                with st.spinner("Thinking…"):
                    response = model.generate_content(context)
                    st.markdown(f"""
                    <div class="bz-card" style="border-left:4px solid #2563EB;">
                        <div class="bz-section-title">🤖 Gemini Response</div>
                        <div style="color:#374151;line-height:1.7;">{response.text}</div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.warning("GOOGLE_API_KEY not found in secrets.")
    except ImportError:
        st.warning("google-generativeai not installed.")
