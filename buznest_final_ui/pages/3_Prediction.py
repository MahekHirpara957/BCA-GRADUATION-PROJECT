import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import inject_theme, check_login, init_connection, page_header

st.set_page_config(page_title="BI Predictions — BuzNet", page_icon="🔮", layout="wide")
inject_theme()
check_login()

supabase = init_connection()

BLUE   = "#2563EB"
LBLUE  = "#3B82F6"
GREEN  = "#10B981"
AMBER  = "#F59E0B"
INDIGO = "#6366F1"

# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_data():
    try:
        res = supabase.table("buznet_data").select("*").eq("client_id", st.session_state["username"]).execute()
        if res.data:
            df = pd.DataFrame(res.data)
            df['Date'] = pd.to_datetime(df['Date'])
            for c in ['Production', 'Sold', 'Revenue']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            return df
        return pd.DataFrame()
    except:
        return pd.DataFrame()

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def make_features(df):
    """
    Date-based feature engineering.
    We extract time components so models can learn seasonal & trend patterns.
    """
    df = df.copy()
    df['dayofweek']  = df['Date'].dt.dayofweek   # 0=Mon, 6=Sun
    df['month']      = df['Date'].dt.month        # 1–12 seasonality
    df['quarter']    = df['Date'].dt.quarter      # Q1–Q4 business cycles
    df['year']       = df['Date'].dt.year         # long-term growth trend
    df['dayofyear']  = df['Date'].dt.dayofyear    # yearly position
    df['dayofmonth'] = df['Date'].dt.day          # within-month pattern
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
    return df

FEATURES = ['dayofweek', 'month', 'quarter', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']

# ── PAGE HEADER ───────────────────────────────────────────────────────────────
page_header("🔮", "Business Intelligence Prediction System",
            "Sales • Revenue • Production — Each powered by the right ML model")

# ── MODEL STRATEGY EXPLANATION ────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#EFF6FF,#F0FDF4);border-left:5px solid #2563EB;
     border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1.5rem;">
    <b style="font-size:1.05rem;">🧠 Smart BI Strategy — Each model is used for different business logic to simulate real BI systems.</b>
    <table style="width:100%;margin-top:0.8rem;border-collapse:collapse;font-size:0.93rem;">
        <tr style="background:#DBEAFE;border-radius:8px;">
            <th style="padding:8px 12px;text-align:left;">ML Model</th>
            <th style="padding:8px 12px;text-align:left;">Business Purpose</th>
            <th style="padding:8px 12px;text-align:left;">Why This Model?</th>
        </tr>
        <tr style="border-bottom:1px solid #E5E7EB;">
            <td style="padding:8px 12px;"><b>📈 Linear Regression</b></td>
            <td style="padding:8px 12px;">Sales Prediction</td>
            <td style="padding:8px 12px;">Best for capturing continuous growth trends over time</td>
        </tr>
        <tr style="background:#F9FAFB;border-bottom:1px solid #E5E7EB;">
            <td style="padding:8px 12px;"><b>🌲 Random Forest</b></td>
            <td style="padding:8px 12px;">Revenue Forecast</td>
            <td style="padding:8px 12px;">Handles complex, nonlinear seasonal & multi-factor data</td>
        </tr>
        <tr>
            <td style="padding:8px 12px;"><b>🌿 Decision Tree</b></td>
            <td style="padding:8px 12px;">Production Planning</td>
            <td style="padding:8px 12px;">Rule-based decisions that mirror real supply-chain logic</td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)

# ── LOAD DATA ────────────────────────────────────────────────────────────────
df = load_data()
if df.empty:
    st.markdown("""<div class="bz-card" style="text-align:center;padding:3rem;">
        <h2>No data available</h2><p>Add records via Data Intake first.</p></div>""",
        unsafe_allow_html=True)
    st.stop()

products = sorted(df['Product'].unique()) if 'Product' in df.columns else []
if not products:
    st.warning("No products found.")
    st.stop()

# ── CONFIGURE ─────────────────────────────────────────────────────────────────
st.markdown('<div class="bz-section-title">📦 Configure Forecast</div>', unsafe_allow_html=True)

sc1, sc2, sc3 = st.columns(3)
selected_product = sc1.selectbox("Select Product", products)
forecast_period  = sc2.selectbox("Forecast Horizon", ["3 Months", "6 Months", "12 Months"])
safety_pct       = sc3.slider("Safety Stock % (for Production)", 0, 50, 10)

profit_margin = st.slider("Profit Margin %", 1, 100, 20)

period_map   = {"3 Months": 90, "6 Months": 180, "12 Months": 365}
horizon_days = period_map[forecast_period]

p_df = df[df['Product'] == selected_product].sort_values('Date').copy()

if len(p_df) < 10:
    st.warning(f"⚠️ Need at least 10 records for **{selected_product}**. Currently: {len(p_df)}")
    st.stop()

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
p_df_feat = make_features(p_df)
X = p_df_feat[FEATURES]

# Split: 80% train, 20% test (time-series order preserved)
split_idx  = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

# ── DEFINE MODELS (ONE PER TARGET) ────────────────────────────────────────────
model_sold       = LinearRegression()          # Sales — trend-based
model_revenue    = RandomForestRegressor(n_estimators=100, random_state=42)   # Revenue — nonlinear
model_production = DecisionTreeRegressor(max_depth=6, random_state=42)        # Production — rule-based

y_sold       = p_df_feat['Sold'].fillna(0)
y_revenue    = p_df_feat['Revenue'].fillna(0)
y_production = p_df_feat['Production'].fillna(0)

y_train_sold,  y_test_sold  = y_sold.iloc[:split_idx],  y_sold.iloc[split_idx:]
y_train_rev,   y_test_rev   = y_revenue.iloc[:split_idx], y_revenue.iloc[split_idx:]
y_train_prod,  y_test_prod  = y_production.iloc[:split_idx], y_production.iloc[split_idx:]

model_sold.fit(X_train, y_train_sold)
model_revenue.fit(X_train, y_train_rev)
model_production.fit(X_train, y_train_prod)

# ── FUTURE DATES & FEATURES ───────────────────────────────────────────────────
last_date    = p_df['Date'].max()
future_dates = [last_date + timedelta(days=x) for x in range(1, horizon_days + 1)]
future_feat  = make_features(pd.DataFrame({'Date': future_dates}))[FEATURES]

pred_sold       = np.clip(model_sold.predict(future_feat), 0, None)
pred_revenue    = np.clip(model_revenue.predict(future_feat), 0, None)
pred_production = np.clip(model_production.predict(future_feat) * (1 + safety_pct / 100), 0, None)

# ── BUSINESS SUMMARY ─────────────────────────────────────────────────────────
total_sales      = pred_sold.sum()
total_revenue    = pred_revenue.sum()
total_production = pred_production.sum()
total_profit     = total_revenue * (profit_margin / 100)

st.markdown("---")
st.markdown('<div class="bz-section-title">🎯 Business Intelligence Summary</div>', unsafe_allow_html=True)

kc1, kc2, kc3, kc4 = st.columns(4)
for col, icon, label, value, model_tag, color in [
    (kc1, "📈", "Sales Forecast",      f"{total_sales:,.0f} units",  "Linear Regression", BLUE),
    (kc2, "💰", "Revenue Forecast",    f"₹{total_revenue:,.0f}",     "Random Forest",     GREEN),
    (kc3, "🏭", "Production Required", f"{total_production:,.0f} units", "Decision Tree", INDIGO),
    (kc4, "💵", "Profit Forecast",     f"₹{total_profit:,.0f}",      f"@ {profit_margin}% margin", AMBER),
]:
    col.markdown(f"""
    <div class="bz-kpi" style="text-align:center;border-top:4px solid {color};">
        <div class="icon">{icon}</div>
        <div class="label">{label}</div>
        <div class="value" style="font-size:1.25rem;color:{color};">{value}</div>
        <div class="delta" style="font-size:0.75rem;margin-top:4px;
             background:{color}18;color:{color};padding:2px 8px;border-radius:20px;
             display:inline-block;">{model_tag}</div>
    </div>""", unsafe_allow_html=True)

# ── FORECAST CHART ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f'<div class="bz-section-title">📅 {forecast_period} Forecast Chart — {selected_product}</div>',
            unsafe_allow_html=True)

# Monthly aggregation for cleaner chart
def monthly(dates, values):
    tmp = pd.DataFrame({'Date': dates, 'Value': values})
    return tmp.set_index('Date').resample('ME')['Value'].sum().reset_index()

hist_sold_m = p_df.set_index('Date').resample('ME')['Sold'].sum().reset_index()
hist_rev_m  = p_df.set_index('Date').resample('ME')['Revenue'].sum().reset_index()
hist_prod_m = p_df.set_index('Date').resample('ME')['Production'].sum().reset_index()

fore_sold_m = monthly(future_dates, pred_sold)
fore_rev_m  = monthly(future_dates, pred_revenue)
fore_prod_m = monthly(future_dates, pred_production)

fig = go.Figure()

# Historical lines
fig.add_trace(go.Scatter(x=hist_sold_m['Date'], y=hist_sold_m['Sold'],
    mode='lines+markers', name='Actual Sales', line=dict(color=BLUE, width=2.5),
    marker=dict(size=5)))
fig.add_trace(go.Scatter(x=hist_rev_m['Date'], y=hist_rev_m['Revenue'],
    mode='lines+markers', name='Actual Revenue', line=dict(color=GREEN, width=2.5),
    marker=dict(size=5)))
fig.add_trace(go.Scatter(x=hist_prod_m['Date'], y=hist_prod_m['Production'],
    mode='lines+markers', name='Actual Production', line=dict(color=INDIGO, width=2.5),
    marker=dict(size=5)))

# Forecast lines
fig.add_trace(go.Scatter(x=fore_sold_m['Date'], y=fore_sold_m['Value'],
    mode='lines+markers', name='📈 Sales Forecast (LR)',
    line=dict(color=BLUE, width=2.5, dash='dot'), marker=dict(size=5, symbol='diamond')))
fig.add_trace(go.Scatter(x=fore_rev_m['Date'], y=fore_rev_m['Value'],
    mode='lines+markers', name='🌲 Revenue Forecast (RF)',
    line=dict(color=GREEN, width=2.5, dash='dot'), marker=dict(size=5, symbol='diamond')))
fig.add_trace(go.Scatter(x=fore_prod_m['Date'], y=fore_prod_m['Value'],
    mode='lines+markers', name='🌿 Production Plan (DT)',
    line=dict(color=INDIGO, width=2.5, dash='dot'), marker=dict(size=5, symbol='diamond')))

# Today line
today_str = str(last_date)[:10]
fig.add_shape(type="line",
    x0=today_str, x1=today_str, y0=0, y1=1,
    xref="x", yref="paper",
    line=dict(color=AMBER, width=2, dash="dash"))
fig.add_annotation(x=today_str, y=0.98,
    xref="x", yref="paper",
    text="▶ Forecast Starts", showarrow=False,
    yanchor="top", font=dict(color=AMBER, size=11))

fig.update_layout(
    title=f"All Predictions — {selected_product} ({forecast_period})",
    plot_bgcolor='white', paper_bgcolor='white',
    hovermode='x unified',
    font=dict(family='Inter, sans-serif'),
    legend=dict(orientation='h', y=-0.18),
    margin=dict(t=50, b=60))
st.plotly_chart(fig, use_container_width=True)

# ── 30-DAY TABLE ──────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 View 30-Day Detailed Prediction Table"):
    tbl = pd.DataFrame({
        'Date':                        [d.strftime('%A, %d %B %Y') for d in future_dates[:30]],
        '📈 Sales (Linear Reg.)':      [int(x) for x in pred_sold[:30]],
        '💰 Revenue (Random Forest)':  [f"₹{x:,.0f}" for x in pred_revenue[:30]],
        '🏭 Production (Decision Tree)': [int(x) for x in pred_production[:30]],
        '💵 Est. Profit':              [f"₹{x * profit_margin / 100:,.0f}" for x in pred_revenue[:30]],
    })
    st.dataframe(tbl, hide_index=True, use_container_width=True)

# ── VIVA EXPLAINER ────────────────────────────────────────────────────────────
with st.expander("🎓 Model Explanation (for Viva)"):
    st.markdown(f"""
### Why Different Models for Different Tasks?

This is a **Business Intelligence System**, not just a basic ML predictor.
Real BI systems choose the right tool for each business problem.

| Task | Model Used | Training Samples | Reason |
|------|-----------|-----------------|--------|
| Sales Prediction | **Linear Regression** | {split_idx} | Sales follow a continuous growth trend — LR captures this simply and explainably |
| Revenue Forecast | **Random Forest** | {split_idx} | Revenue is affected by many factors (seasons, holidays, discounts) — RF handles this complexity |
| Production Planning | **Decision Tree** | {split_idx} | Production uses threshold-based rules (if demand > X, produce Y) — DT mirrors this logic |

**Feature Engineering Used:**
- Day of week, Month, Quarter → captures seasonal patterns
- Year → captures long-term growth
- Day of year, Week of year → captures cyclical business patterns

**Safety Stock ({safety_pct}%)** is added on top of predicted production
to ensure supply never runs short.

> *Viva Answer:* "We designed this as a Business Intelligence System where each ML model is
> assigned to the business task it is best suited for. Linear Regression for sales trends,
> Random Forest for complex revenue patterns, and Decision Tree for rule-based production
> planning — this mirrors how real enterprise BI systems work."
    """)
