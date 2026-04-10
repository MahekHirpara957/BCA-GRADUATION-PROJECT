import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import inject_theme, check_login, init_connection, page_header

st.set_page_config(page_title="Dashboard — BuzNet", page_icon="📊", layout="wide")
inject_theme()
check_login()

supabase = init_connection()
BLUE = "#2563EB"; LBLUE = "#3B82F6"; GREEN = "#10B981"; INDIGO = "#6366F1"; AMBER = "#F59E0B"
COLORS = ["#2563EB","#3B82F6","#10B981","#6366F1","#F59E0B","#14B8A6","#8B5CF6","#EF4444"]

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

page_header("📊", "Analytics Dashboard", "Industry-level business performance visualisation & intelligence")

df = load_data()

if df.empty:
    st.markdown("""
    <div class="bz-card" style="text-align:center;padding:3rem;">
        <div style="font-size:4rem;margin-bottom:1rem;">📊</div>
        <h2 style="color:#1F2937;">No Data Yet</h2>
        <p style="color:#6B7280;">Go to <strong>Data Intake</strong> to add your first records.</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── KPI CARDS ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">📌 Key Performance Indicators</div>', unsafe_allow_html=True)

total_rev  = df['Revenue'].sum()
total_sold = df['Sold'].sum()
total_prod = df['Production'].sum()
sold_rate  = (total_sold / total_prod * 100) if total_prod > 0 else 0
avg_price  = (total_rev / total_sold) if total_sold > 0 else 0

now    = df['Date'].max()
prev   = now - pd.DateOffset(months=1)
this_m = df[df['Date'].dt.to_period('M') == now.to_period('M')]['Revenue'].sum()
prev_m = df[df['Date'].dt.to_period('M') == prev.to_period('M')]['Revenue'].sum()
mom    = ((this_m - prev_m) / prev_m * 100) if prev_m > 0 else 0

k1, k2, k3, k4, k5 = st.columns(5, gap="medium")
for col, icon, label, value, delta in [
    (k1, "💰", "Total Revenue",    f"₹{total_rev:,.0f}",  f"{mom:+.1f}% MoM"),
    (k2, "📦", "Total Units Sold", f"{total_sold:,.0f}",  f"{sold_rate:.1f}% sell-through"),
    (k3, "🏭", "Total Production", f"{total_prod:,.0f}",  f"{df['Product'].nunique()} products"),
    (k4, "📈", "This Month Rev",   f"₹{this_m:,.0f}",    f"vs ₹{prev_m:,.0f} last month"),
    (k5, "💵", "Avg Unit Price",   f"₹{avg_price:,.2f}", "revenue / units sold"),
]:
    col.markdown(f"""
    <div class="bz-kpi">
        <div class="icon">{icon}</div>
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="delta">{delta}</div>
    </div>""", unsafe_allow_html=True)

# ── GOAL TRACKER ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">🎯 Revenue Goal Tracker</div>', unsafe_allow_html=True)

gc1, gc2, gc3 = st.columns([1, 2, 1])
with gc2:
    goal = st.number_input("Set Revenue Target (₹):", min_value=1.0, value=100000.0, step=5000.0)

pct = min(total_rev / goal, 1.0)
st.progress(pct)
if pct >= 1.0:
    st.success(f"🎉 Goal crushed! **{pct*100:.1f}%** achieved — ₹{total_rev:,.0f} of ₹{goal:,.0f}")
elif pct >= 0.75:
    st.info(f"🚀 Almost there! **{pct*100:.1f}%** — ₹{goal - total_rev:,.0f} remaining")
elif pct >= 0.5:
    st.info(f"💪 Good progress! **{pct*100:.1f}%** — Keep pushing!")
else:
    st.info(f"📈 Getting started: **{pct*100:.1f}%** — Every entry counts!")

# ── TREND ANALYSIS ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">📈 Trend Analysis</div>', unsafe_allow_html=True)

time_basis = st.radio("Period:", ["Daily","Weekly","Monthly","Quarterly","Yearly"], horizontal=True)
freq_map   = {"Daily":"D","Weekly":"W","Monthly":"ME","Quarterly":"QE","Yearly":"YE"}
resampled  = df.set_index('Date').resample(freq_map[time_basis])[['Revenue','Production','Sold']].sum().reset_index()

tc1, tc2 = st.columns(2, gap="large")
with tc1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=resampled['Date'], y=resampled['Revenue'],
        mode='lines+markers', name='Revenue',
        line=dict(color=BLUE, width=3), marker=dict(size=7),
        fill='tozeroy', fillcolor='rgba(37,99,235,.08)'))
    fig.update_layout(title=f"Revenue Trend ({time_basis})", plot_bgcolor='white',
        paper_bgcolor='white', hovermode='x unified', font=dict(family='Inter'))
    st.plotly_chart(fig, use_container_width=True)

with tc2:
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=resampled['Date'], y=resampled['Sold'],
        name='Units Sold', marker_color=LBLUE, opacity=0.85))
    fig2.update_layout(title=f"Units Sold ({time_basis})", plot_bgcolor='white',
        paper_bgcolor='white', font=dict(family='Inter'))
    st.plotly_chart(fig2, use_container_width=True)

# ── THIS MONTH vs LAST MONTH ──────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">📅 This Month vs Last Month</div>', unsafe_allow_html=True)

mc1, mc2 = st.columns(2, gap="large")
with mc1:
    comp = pd.DataFrame({'Period': ['Last Month','This Month'], 'Revenue': [prev_m, this_m]})
    fig_comp = px.bar(comp, x='Period', y='Revenue', color='Period',
        color_discrete_map={'Last Month':'#BFDBFE','This Month':BLUE},
        text_auto=True, title="Revenue Comparison")
    fig_comp.update_layout(plot_bgcolor='white', paper_bgcolor='white',
        showlegend=False, font=dict(family='Inter'))
    st.plotly_chart(fig_comp, use_container_width=True)

with mc2:
    prod_rev = df.groupby('Product')['Revenue'].sum().reset_index().sort_values('Revenue', ascending=False)
    fig_prod = px.bar(prod_rev, x='Product', y='Revenue', title="Revenue by Product",
        color='Revenue', color_continuous_scale=['#EFF6FF', BLUE], text_auto=True)
    fig_prod.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Inter'))
    st.plotly_chart(fig_prod, use_container_width=True)

# ── TOP / BOTTOM ANALYSIS ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">🏆 Top & Bottom Product Analysis</div>', unsafe_allow_html=True)

prod_perf = df.groupby('Product').agg(
    Revenue=('Revenue','sum'), Sold=('Sold','sum'),
    Production=('Production','sum')).reset_index().sort_values('Revenue', ascending=False)

ab1, ab2 = st.columns(2, gap="large")
with ab1:
    top5 = prod_perf.head(5)
    fig_top = px.bar(top5, y='Product', x='Revenue', orientation='h',
        color='Revenue', color_continuous_scale=['#BFDBFE', BLUE],
        text_auto=True, title="🥇 Top 5 Products by Revenue")
    fig_top.update_layout(plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Inter'), showlegend=False, yaxis=dict(autorange='reversed'))
    st.plotly_chart(fig_top, use_container_width=True)

with ab2:
    bot5 = prod_perf.tail(5).sort_values('Revenue')
    fig_bot = px.bar(bot5, y='Product', x='Revenue', orientation='h',
        color='Revenue', color_continuous_scale=['#BFDBFE', LBLUE],
        text_auto=True, title="🔻 Bottom 5 Products (Need Attention)")
    fig_bot.update_layout(plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Inter'), showlegend=False, yaxis=dict(autorange='reversed'))
    st.plotly_chart(fig_bot, use_container_width=True)

# ── PRODUCT DRILL-DOWN ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">🔍 Product Drill-Down</div>', unsafe_allow_html=True)

selected = st.selectbox("Select Product for Detailed Analysis", sorted(df['Product'].unique()))
pdd = df[df['Product'] == selected]

dd1, dd2, dd3, dd4 = st.columns(4)
dd1.metric("Total Revenue",   f"₹{pdd['Revenue'].sum():,.0f}")
dd2.metric("Total Sold",      f"{pdd['Sold'].sum():,.0f}")
dd3.metric("Total Produced",  f"{pdd['Production'].sum():,.0f}")
dd4.metric("Avg Unit Price",  f"₹{(pdd['Revenue'].sum()/max(pdd['Sold'].sum(),1)):,.2f}")

pdd_monthly = pdd.set_index('Date').resample('ME')[['Revenue','Sold','Production']].sum().reset_index()
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=pdd_monthly['Date'], y=pdd_monthly['Revenue'],
    name='Revenue', line=dict(color=BLUE, width=3), mode='lines+markers'))
fig_dd.add_trace(go.Bar(x=pdd_monthly['Date'], y=pdd_monthly['Sold'],
    name='Units Sold', marker_color=LBLUE, opacity=0.6, yaxis='y2'))
fig_dd.update_layout(
    title=f"Monthly Performance — {selected}",
    plot_bgcolor='white', paper_bgcolor='white', hovermode='x unified',
    font=dict(family='Inter'),
    yaxis=dict(title='Revenue (₹)'),
    yaxis2=dict(title='Units Sold', overlaying='y', side='right'),
    legend=dict(orientation='h', y=-0.15))
st.plotly_chart(fig_dd, use_container_width=True)

with st.expander(f"📋 View All Records for {selected}"):
    show = pdd.drop(columns=['client_id'], errors='ignore').copy()
    show['Date'] = show['Date'].dt.strftime('%Y-%m-%d')
    st.dataframe(show.sort_values('Date', ascending=False), hide_index=True, use_container_width=True)

# ── PIE CHARTS ────────────────────────────────────────────────────────────────
st.markdown("---")
pc1, pc2 = st.columns(2, gap="large")
with pc1:
    fig_pie = px.pie(prod_perf, names='Product', values='Revenue',
        title="Revenue Share by Product", hole=0.4, color_discrete_sequence=['#2563EB','#3B82F6','#10B981','#6366F1','#F59E0B','#14B8A6','#8B5CF6','#EF4444'])
    fig_pie.update_layout(paper_bgcolor='white', font=dict(family='Inter'))
    st.plotly_chart(fig_pie, use_container_width=True)

with pc2:
    fig_pie2 = px.pie(prod_perf, names='Product', values='Sold',
        title="Sales Share by Product", hole=0.4, color_discrete_sequence=['#2563EB','#3B82F6','#10B981','#6366F1','#F59E0B','#14B8A6','#8B5CF6','#EF4444'])
    fig_pie2.update_layout(paper_bgcolor='white', font=dict(family='Inter'))
    st.plotly_chart(fig_pie2, use_container_width=True)

# ── HEATMAP ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">🌡️ Revenue Heatmap (Product × Month)</div>', unsafe_allow_html=True)

hm_df = df.copy()
hm_df['Month'] = hm_df['Date'].dt.to_period('M').astype(str)
heat = hm_df.groupby(['Product','Month'])['Revenue'].sum().reset_index()
heat_pivot = heat.pivot(index='Product', columns='Month', values='Revenue').fillna(0)
fig_heat = px.imshow(heat_pivot, color_continuous_scale=['#F8FAFC','#BFDBFE', BLUE],
    title="Revenue Heatmap (Product × Month)",
    labels=dict(color="Revenue (₹)"), aspect="auto", text_auto=".0f")
fig_heat.update_layout(paper_bgcolor='white', font=dict(family='Inter'))
st.plotly_chart(fig_heat, use_container_width=True)
