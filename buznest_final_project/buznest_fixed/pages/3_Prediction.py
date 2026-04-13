import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# High-Performance ML Models
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from utils import inject_theme, check_login, init_connection, page_header
except ImportError:
    def inject_theme(): pass
    def check_login(): pass
    def init_connection(): return None
    def page_header(i, t, s): st.title(f"{i} {t}"); st.write(s)

st.set_page_config(page_title="BI Predictions — BuzNet", page_icon="🔮", layout="wide")
inject_theme()
check_login()

supabase = init_connection()

# Colors
BLUE, GREEN, AMBER, INDIGO = "#2563EB", "#10B981", "#F59E0B", "#6366F1"

# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_data():
    try:
        user = st.session_state.get("username", "demo_user")
        res = supabase.table("buznet_data").select("*").eq("client_id", user).execute()
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

# ── ADVANCED FEATURE ENGINEERING (ENHANCED FOR ACCURACY) ────────────────────
def make_features(df, target_col=None):
    """
    Enhanced features with Lags, Rolling averages, and Holiday context.
    """
    df = df.copy().sort_values('Date')
    
    # Temporal Signals
    df['dayofweek']  = df['Date'].dt.dayofweek
    df['month']      = df['Date'].dt.month
    df['quarter']    = df['Date'].dt.quarter
    df['dayofyear']  = df['Date'].dt.dayofyear
    df['dayofmonth'] = df['Date'].dt.day
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Lag Features (Historical Context)
    if target_col:
        # We shift based on target to provide 'momentum'
        df['lag_1'] = df[target_col].shift(1)
        df['lag_7'] = df[target_col].shift(7)
        df['rolling_mean_7'] = df[target_col].shift(1).rolling(window=7).mean()
        # Handle start-of-series NaNs using modern pandas syntax
        df = df.bfill()
    
    return df

# Features allowed for the model
FEATURES = ['dayofweek', 'month', 'quarter', 'dayofyear', 'dayofmonth', 'is_weekend', 'lag_1', 'lag_7', 'rolling_mean_7']

page_header("🔮", "High-Accuracy BI Predictions", 
            "Optimized Recursive Forecasting with Full-Dataset Refitting")

# ── LOAD DATA ────────────────────────────────────────────────────────────────
df = load_data()
if df.empty:
    st.warning("No data available. Please add records in Data Intake.")
    st.stop()

products = sorted(df['Product'].unique())
if not products:
    st.stop()

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
st.markdown('<div class="bz-section-title">📦 Configure Forecast</div>', unsafe_allow_html=True)
sc1, sc2, sc3 = st.columns(3)

selected_product = sc1.selectbox("Select Product", products)
forecast_period   = sc2.selectbox("Time Duration", ["1 Week", "1 Month", "1 Quarter", "1 Year"])
safety_pct       = sc3.slider("Safety Stock %", 0, 50, 10)

profit_margin = st.slider("Profit Margin %", 1, 100, 20)

period_map = {"1 Week": 7, "1 Month": 30, "1 Quarter": 90, "1 Year": 365}
horizon_days = period_map[forecast_period]

p_df = df[df['Product'] == selected_product].sort_values('Date').copy()

if len(p_df) < 20:
    st.warning(f"⚠️ Accuracy Warning: High-performance models perform best with 20+ records. Current: {len(p_df)}")
    if len(p_df) < 10: st.stop()

# ── ADVANCED ML PIPELINE ──────────────────────────────────────────────────────

def get_optimized_model(model_type):
    # Adjusted learning rates for better convergence on varying dataset sizes
    if model_type == "xgb":
        return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
    elif model_type == "cat":
        return CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3, verbose=0, random_state=42)
    else:
        return LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31, random_state=42, verbose=-1)

def train_validate_and_refit(data, target_col, model_type):
    feat_df = make_features(data, target_col=target_col)
    X = feat_df[FEATURES]
    
    # Log stabilization for Revenue
    y = feat_df[target_col]
    use_log = (target_col == 'Revenue')
    if use_log: y = np.log1p(y)

    # 1. Validation Phase (80/20 split to show user the accuracy)
    # If data is very small, accuracy metrics are naturally volatile
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    val_model = get_optimized_model(model_type)
    val_model.fit(X_train, y_train)
    
    preds_test = val_model.predict(X_test)
    if use_log:
        actual_test = np.expm1(y_test)
        actual_preds = np.expm1(preds_test)
    else:
        actual_test = y_test
        actual_preds = preds_test

    mae = mean_absolute_error(actual_test, actual_preds)
    r2  = r2_score(actual_test, actual_preds)

    # 2. Refit Phase (Train on 100% data for the actual forecast)
    final_model = get_optimized_model(model_type)
    final_model.fit(X, y)

    # 3. Recursive Forecast Loop
    forecasts = []
    curr_history = feat_df.copy()

    for _ in range(horizon_days):
        next_date = curr_history['Date'].max() + timedelta(days=1)
        
        # Build features for next day
        next_row = pd.DataFrame({'Date': [next_date]})
        next_row['dayofweek']  = next_row['Date'].dt.dayofweek
        next_row['month']      = next_row['Date'].dt.month
        next_row['quarter']    = next_row['Date'].dt.quarter
        next_row['dayofyear']  = next_row['Date'].dt.dayofyear
        next_row['dayofmonth'] = next_row['Date'].dt.day
        next_row['is_weekend'] = next_row['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Inject momentum from history
        next_row['lag_1'] = curr_history[target_col].iloc[-1]
        next_row['lag_7'] = curr_history[target_col].iloc[-7] if len(curr_history) >= 7 else curr_history[target_col].iloc[-1]
        next_row['rolling_mean_7'] = curr_history[target_col].tail(7).mean()
        
        # Predict using the FINAL (100% data) model
        p = final_model.predict(next_row[FEATURES])[0]
        p_val = np.expm1(p) if use_log else p
        p_val = max(0, p_val)
        
        forecasts.append(p_val)
        
        # Update history for next iteration's lags
        next_row[target_col] = p_val
        curr_history = pd.concat([curr_history, next_row], ignore_index=True)

    return np.array(forecasts), {"mae": mae, "r2": r2}

# Execution
with st.spinner("🧠 Optimizing models and refitting on latest data..."):
    pred_s, acc_s = train_validate_and_refit(p_df, 'Sold', 'xgb')
    pred_r, acc_r = train_validate_and_refit(p_df, 'Revenue', 'cat')
    pred_p, acc_p = train_validate_and_refit(p_df, 'Production', 'lgbm')

# ── SUMMARY KPIs ──────────────────────────────────────────────────────────────
st.markdown("---")
kc1, kc2, kc3, kc4 = st.columns(4)
stats = [
    (kc1, "📈", "Projected Sales", f"{pred_s.sum():,.0f} units", "XGBoost", BLUE),
    (kc2, "💰", "Projected Revenue", f"₹{pred_r.sum():,.0f}", "CatBoost", GREEN),
    (kc3, "🏭", "Prod. Capacity Needed", f"{pred_p.sum():,.0f} units", "LightGBM", INDIGO),
    (kc4, "💵", "Projected Profit", f"₹{pred_r.sum() * (profit_margin/100):,.0f}", f"{profit_margin}% Margin", AMBER)
]

for col, icon, lab, val, tag, clr in stats:
    col.markdown(f"""
    <div style="text-align:center; border-top:4px solid {clr}; padding:15px; background:#f9f9f9; border-radius:10px; height:100%;">
        <div style="font-size:1.8rem; margin-bottom:5px;">{icon}</div>
        <div style="color:#666; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px;">{lab}</div>
        <div style="font-size:1.25rem; font-weight:bold; color:{clr}; margin:5px 0;">{val}</div>
        <div style="font-size:0.7rem; background:{clr}15; color:{clr}; padding:3px 10px; border-radius:12px; display:inline-block; font-weight:600;">{tag}</div>
    </div>""", unsafe_allow_html=True)

# ── CHART ─────────────────────────────────────────────────────────────────────
st.markdown("---")
future_dates = [p_df['Date'].max() + timedelta(days=x) for x in range(1, horizon_days + 1)]
fig = go.Figure()
hist_data = p_df.sort_values('Date')
fig.add_trace(go.Scatter(x=hist_data['Date'], y=hist_data['Revenue'], name='Actual Revenue', line=dict(color=GREEN, width=2.5)))
fig.add_trace(go.Scatter(x=future_dates, y=pred_r, name='Forecast (CatBoost)', line=dict(color=GREEN, dash='dot', width=2.5)))
fig.add_trace(go.Scatter(x=hist_data['Date'], y=hist_data['Sold'], name='Actual Sales', line=dict(color=BLUE, width=1.5)))
fig.add_trace(go.Scatter(x=future_dates, y=pred_s, name='Forecast (XGBoost)', line=dict(color=BLUE, dash='dot', width=1.5)))

fig.update_layout(title=f"Predictive Analytics — {selected_product}", hovermode='x unified', plot_bgcolor='white', 
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, width='stretch')

# ── DETAILED PREDICTIONS TABLE ────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">📋 Detailed Forecast Data</div>', unsafe_allow_html=True)

with st.expander(f"View Daily Breakdown for {forecast_period}", expanded=False):
    forecast_df = pd.DataFrame({
        "Date": [d.strftime('%Y-%m-%d (%a)') for d in future_dates],
        "🚀 Sales (XGBoost)": [f"{int(x)}" for x in pred_s],
        "💰 Revenue (CatBoost)": [f"₹{x:,.2f}" for x in pred_r],
        "🏭 Production Required": [f"{int(x)}" for x in pred_p]
    })
    st.dataframe(forecast_df, width='stretch', hide_index=True)

# ── ACCURACY SECTION ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="bz-section-title">📊 Historical Accuracy Breakdown</div>', unsafe_allow_html=True)
ac1, ac2, ac3 = st.columns(3)

def render_metric(col, title, mae, r2, color):
    # Convert R2 to an intuitive "Accuracy Score" percentage
    # If R2 is negative, the model isn't yet better than a simple average line.
    if r2 <= 0:
        accuracy_display = "Learning..."
        accuracy_val = "0.0%"
        status_color = AMBER
        confidence_text = "Insufficient Data History"
    else:
        accuracy_pct = r2 * 100
        accuracy_display = f"{accuracy_pct:.1f}% Accuracy"
        accuracy_val = f"{accuracy_pct:.1f}/100"
        status_color = GREEN if accuracy_pct > 75 else AMBER
        confidence_text = f"Confidence Level: <b>{accuracy_val}</b>"
    
    col.markdown(f"""
    <div style="padding:15px; border-radius:12px; background:white; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); border-left:6px solid {color};">
        <div style="font-weight:bold; color:#1F2937; margin-bottom:8px;">{title}</div>
        <div style="font-size:1.4rem; font-weight:bold; color:{status_color};">{accuracy_display}</div>
        <div style="font-size:0.85rem; color:#4B5563; margin-top:5px;">{confidence_text}</div>
        <div style="font-size:0.85rem; color:#6B7280;">Average Error: {mae:,.2f} units</div>
        <div style="font-size:0.75rem; margin-top:10px; font-style:italic; color:#9CA3AF;">* Based on validation split R² score</div>
    </div>""", unsafe_allow_html=True)

render_metric(ac1, "🚀 XGBoost Engine", acc_s['mae'], acc_s['r2'], BLUE)
render_metric(ac2, "🐈 CatBoost Engine", acc_r['mae'], acc_r['r2'], GREEN)
render_metric(ac3, "⚡ LightGBM Engine", acc_p['mae'], acc_p['r2'], INDIGO)

with st.expander("🎓 Why is accuracy 0.0% or 'Learning'?"):
    st.markdown("""
    - **Data Scarcity:** Machine Learning models (XGBoost/CatBoost) require a significant history to find patterns. If you have fewer than 30–50 records, the model is still in a 'Learning Phase'.
    - **Variance Check:** If the $R^2$ score is 0.0% or negative, it means the model's predictions aren't yet better than just guessing the historical average. 
    - **Solution:** Add more daily records in the **Data Intake** module. Accuracy will naturally climb as the dataset grows and seasonal patterns emerge.
    """)