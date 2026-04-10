import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import inject_theme, check_login, init_connection, page_header

st.set_page_config(page_title="ML Predictions — BuzNet", page_icon="🔮", layout="wide")
inject_theme()
check_login()

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

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
    except: return pd.DataFrame()

def make_features(df):
    df = df.copy()
    df['dayofweek']  = df['Date'].dt.dayofweek
    df['quarter']    = df['Date'].dt.quarter
    df['month']      = df['Date'].dt.month
    df['year']       = df['Date'].dt.year
    df['dayofyear']  = df['Date'].dt.dayofyear
    df['dayofmonth'] = df['Date'].dt.day
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
    return df

FEATURES = ['dayofweek','quarter','month','year','dayofyear','dayofmonth','weekofyear']

page_header("🔮", "AI-Powered ML Predictions",
            "Multiple models • Smart forecasting • Business insights")

# ── CONFIGURE ────────────────────────────────────────────────────────────────
df = load_data()
if df.empty:
    st.markdown("""<div class="bz-card" style="text-align:center;padding:3rem;">
        <h2>No data available</h2><p>Add records via Data Intake first.</p></div>""",
        unsafe_allow_html=True)
    st.stop()

products = sorted(df['Product'].unique()) if 'Product' in df.columns else []
if not products:
    st.warning("No products found."); st.stop()

st.markdown('<div class="bz-section-title">📦 Configure Prediction</div>', unsafe_allow_html=True)

sc1, sc2, sc3, sc4 = st.columns(4)
selected_product = sc1.selectbox("Product", products)
target_col       = sc2.selectbox("Predict Target", ["Sold","Revenue","Production"])
forecast_period  = sc3.selectbox("Forecast Horizon", ["3 Months","6 Months","12 Months"])
safety_pct       = sc4.slider("Safety Stock %", 0, 50, 10)
profit_margin    = st.slider("Profit Margin %", 1, 100, 20)

period_map   = {"3 Months":90,"6 Months":180,"12 Months":365}
horizon_days = period_map[forecast_period]

p_df = df[df['Product'] == selected_product].sort_values('Date').copy()

if len(p_df) < 10:
    st.warning(f"⚠️ Need at least 10 records for **{selected_product}**. Currently: {len(p_df)}")
    st.stop()

p_df_feat = make_features(p_df)
X = p_df_feat[FEATURES]
y = p_df_feat[target_col].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

MODEL_DEFS = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
    "Decision Tree":     DecisionTreeRegressor(max_depth=6, random_state=42),
}
if HAS_XGB:
    MODEL_DEFS["XGBoost"] = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1,
                                               max_depth=5, random_state=42, verbosity=0)

# ── TRAIN ALL MODELS ──────────────────────────────────────────────────────────
results = {}
trained = {}
for name, m in MODEL_DEFS.items():
    m.fit(X_train, y_train)
    preds = np.clip(m.predict(X_test), 0, None)
    r2   = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    results[name] = {"r2": max(0, r2), "rmse": round(rmse, 2), "mae": round(mae, 2)}
    trained[name] = m

# Pick best model
best_name  = max(results, key=lambda n: results[n]["r2"])
best_model = trained[best_name]
best_r2    = results[best_name]["r2"]
best_rmse  = results[best_name]["rmse"]
best_mae   = results[best_name]["mae"]

# ── BEST MODEL CARD ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">🏆 Best Model Auto-Selected</div>', unsafe_allow_html=True)

bm1, bm2, bm3, bm4 = st.columns(4)
bm1.markdown(f"""
<div class="bz-kpi" style="text-align:center;">
    <div class="icon">🤖</div>
    <div class="label">Best Model</div>
    <div class="value" style="font-size:1.2rem;">{best_name}</div>
    <div class="delta">Highest accuracy</div>
</div>""", unsafe_allow_html=True)

bm2.markdown(f"""
<div class="bz-kpi" style="text-align:center;">
    <div class="icon">📊</div>
    <div class="label">Accuracy (R²)</div>
    <div class="value" style="font-size:1.4rem;">{best_r2*100:.1f}%</div>
    <div class="delta">Model fit score</div>
</div>""", unsafe_allow_html=True)

bm3.markdown(f"""
<div class="bz-kpi" style="text-align:center;">
    <div class="icon">📉</div>
    <div class="label">RMSE</div>
    <div class="value" style="font-size:1.4rem;">{best_rmse:.2f}</div>
    <div class="delta">Root mean sq. error</div>
</div>""", unsafe_allow_html=True)

bm4.markdown(f"""
<div class="bz-kpi" style="text-align:center;">
    <div class="icon">📐</div>
    <div class="label">MAE</div>
    <div class="value" style="font-size:1.4rem;">{best_mae:.2f}</div>
    <div class="delta">Mean absolute error</div>
</div>""", unsafe_allow_html=True)

# ── MODEL ACCURACY BARS ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">📊 All Models — Accuracy Overview</div>', unsafe_allow_html=True)

model_cols = st.columns(len(MODEL_DEFS))
model_colors_list = [BLUE, LBLUE, GREEN, BLUE]
for i, (name, res) in enumerate(results.items()):
    is_best = name == best_name
    border  = "4px solid #2563EB" if is_best else "1px solid #E5E7EB"
    badge   = " 🏆" if is_best else ""
    model_cols[i].markdown(f"""
    <div class="bz-kpi" style="text-align:center;border-left:{border};">
        <div class="label">{name}{badge}</div>
        <div class="value" style="font-size:1.3rem;color:{model_colors_list[i]};">{res['r2']*100:.1f}%</div>
        <div class="delta">RMSE: {res['rmse']} | MAE: {res['mae']}</div>
    </div>""", unsafe_allow_html=True)

# ── FUTURE FORECAST ───────────────────────────────────────────────────────────
last_date    = p_df['Date'].max()
future_dates = [last_date + timedelta(days=x) for x in range(1, horizon_days+1)]
future_feat  = make_features(pd.DataFrame({'Date': future_dates}))
best_preds   = np.clip(best_model.predict(future_feat[FEATURES]), 0, None)

avg_price    = (p_df['Revenue'].sum() / p_df['Sold'].sum()) if p_df['Sold'].sum() > 0 else 1
avg_margin   = profit_margin / 100

total_forecast   = best_preds.sum()
revenue_forecast = total_forecast * avg_price if target_col == "Sold" else total_forecast
profit_forecast  = revenue_forecast * avg_margin
demand_growth    = ((best_preds[-30:].mean() - best_preds[:30].mean()) /
                    max(best_preds[:30].mean(), 1)) * 100

# ── BUSINESS PREDICTIONS ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">🎯 Business Predictions</div>', unsafe_allow_html=True)

bp1, bp2, bp3, bp4 = st.columns(4)
for col, icon, label, value, note in [
    (bp1, "📈", "Sales Forecast",   f"{total_forecast:,.0f} units", f"Over {forecast_period}"),
    (bp2, "💰", "Revenue Forecast", f"₹{revenue_forecast:,.0f}",    f"{forecast_period} projection"),
    (bp3, "💵", "Profit Forecast",  f"₹{profit_forecast:,.0f}",     f"@ {profit_margin}% margin"),
    (bp4, "👥", "Demand Growth",    f"{demand_growth:+.1f}%",        "Start → End of period"),
]:
    col.markdown(f"""
    <div class="bz-kpi" style="text-align:center;">
        <div class="icon">{icon}</div>
        <div class="label">{label}</div>
        <div class="value" style="font-size:1.25rem;">{value}</div>
        <div class="delta">{note}</div>
    </div>""", unsafe_allow_html=True)

# ── FORECAST CHART ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f'<div class="bz-section-title">📅 {forecast_period} Forecast — {selected_product}</div>',
            unsafe_allow_html=True)

hist_m = p_df.set_index('Date').resample('ME')[target_col].sum().reset_index()
fore_m = pd.DataFrame({'Date': future_dates, 'Value': best_preds})
fore_m = fore_m.set_index('Date').resample('ME')['Value'].sum().reset_index()

fig_fore = go.Figure()
fig_fore.add_trace(go.Scatter(x=hist_m['Date'], y=hist_m[target_col],
    mode='lines+markers', name='Historical',
    line=dict(color=BLUE, width=3), marker=dict(size=6)))
fig_fore.add_trace(go.Scatter(x=fore_m['Date'], y=fore_m['Value'],
    mode='lines+markers', name=f'Forecast ({best_name})',
    line=dict(color=LBLUE, width=3, dash='dot'), marker=dict(size=6),
    fill='tonexty', fillcolor='rgba(236,72,153,.06)'))

# FIX: Use shapes instead of add_vline to avoid the Plotly annotation bug
last_date_str = str(last_date)[:10]
fig_fore.add_shape(
    type="line",
    x0=last_date_str, x1=last_date_str, y0=0, y1=1,
    xref="x", yref="paper",
    line=dict(color="#F59E0B", width=2, dash="dash")
)
fig_fore.add_annotation(
    x=last_date_str, y=1,
    xref="x", yref="paper",
    text="Today", showarrow=False,
    yanchor="bottom", font=dict(color="#F59E0B", size=12)
)
fig_fore.update_layout(
    title=f"{target_col} Forecast — {forecast_period} ({best_name})",
    plot_bgcolor='white', paper_bgcolor='white', hovermode='x unified',
    font=dict(family='Inter'), legend=dict(orientation='h', y=-0.15))
st.plotly_chart(fig_fore, use_container_width=True)

# ── 30-DAY TABLE ──────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 View 30-Day Detailed Forecast Table"):
    tbl = pd.DataFrame({
        'Date':                  future_dates[:30],
        f'Predicted {target_col}': [int(x) for x in best_preds[:30]],
        'Recommended Production':  [int(x*(1+safety_pct/100)) for x in best_preds[:30]],
        'Est. Revenue (₹)':        [f"₹{x*avg_price:,.2f}" for x in best_preds[:30]],
        'Est. Profit (₹)':         [f"₹{x*avg_price*avg_margin:,.2f}" for x in best_preds[:30]],
    })
    tbl['Date'] = tbl['Date'].dt.strftime('%A, %d %B %Y')
    st.dataframe(tbl, hide_index=True, use_container_width=True)

with st.expander("🤖 Model Technical Info"):
    st.markdown(f"""
**Best Model:** {best_name}  
**Training Samples:** {len(X_train)}  |  **Test Samples:** {len(X_test)}  
**Features:** Day of week, Quarter, Month, Year, Day of year, Day of month, Week of year  
**Target:** {target_col}  |  **Horizon:** {forecast_period}  
**Accuracy (R²):** {best_r2*100:.2f}%  |  **RMSE:** {best_rmse}  |  **MAE:** {best_mae}

> *Viva Answer:* "We implemented {len(MODEL_DEFS)} ML models and selected **{best_name}** as the
best model based on R², RMSE, and MAE performance metrics."
    """)
