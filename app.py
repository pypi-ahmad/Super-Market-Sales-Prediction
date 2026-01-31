"""
Universal Forecasting Lab 🔮
A Streamlit dashboard for interactive time-series analysis and forecasting.
Powered by the ForecastingEngine (FLAML).
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from train_forecast import ForecastingEngine

# --- Page Config ---
st.set_page_config(page_title="Research Lab Dashboard", page_icon="🔬", layout="wide")

# --- CSS Styling ---
# Dark mode enhancements for a professional look
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stMetric { background-color: #262730; padding: 15px; border-radius: 10px; }
    h1, h2, h3 { color: #00ADB5; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #262730; border-radius: 4px; color: #fff; } 
    .stTabs [aria-selected="true"] { background-color: #00ADB5; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_resource
def load_bundle(filename='automl_bundle.pkl'):
    """Loads the trained model bundle from disk safely."""
    if os.path.exists(filename):
        try:
            return joblib.load(filename)
        except:
            return None
    return None

def train_model(df, date_col, target_col, time_budget):
    """Initializes and runs the ForecastingEngine."""
    # Reset session state bundle to force reload after training
    if 'bundle' in st.session_state:
        del st.session_state['bundle']
        
    engine = ForecastingEngine(time_budget=time_budget)
    engine.preprocess(df, date_col, target_col)
    engine.train()
    engine.save('automl_bundle.pkl')
    return engine

# --- Sidebar ---
st.sidebar.title("🔬 Research Lab")
data_source = st.sidebar.radio("Data Source", ["Use Demo Data", "Upload CSV"])

df = None
date_col = 'Date'
target_col = 'Total'

if data_source == "Use Demo Data":
    if os.path.exists("supermarket_sales.csv"):
        df = pd.read_csv("supermarket_sales.csv")
        # Ensure correct defaults for demo data
        date_col = 'Date'
        target_col = 'Total'
    else:
        st.error("supermarket_sales.csv not found! Please upload data.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        cols = df.columns.tolist()
        date_col = st.sidebar.selectbox("Select Date Column", cols, index=0)
        target_col = st.sidebar.selectbox("Select Target Column", cols, index=min(1, len(cols)-1))

time_budget = st.sidebar.slider("Time Budget (s)", 10, 300, 30)
train_btn = st.sidebar.button("🚀 Start Experiment")

if train_btn and df is not None:
    with st.spinner(f"Running Experiment (Budget: {time_budget}s)..."):
        try:
            # Clear cache to ensure fresh run
            load_bundle.clear()
            engine = train_model(df, date_col, target_col, time_budget)
            st.success("Experiment Complete! Analysis Ready.")
            # Reload bundle
            st.session_state.bundle = load_bundle()
            st.rerun()
        except Exception as e:
            st.error(f"Training failed: {e}")

# Load bundle from disk if not in session
if 'bundle' not in st.session_state or st.session_state.bundle is None:
    st.session_state.bundle = load_bundle()

# --- Main Dashboard ---
if st.session_state.bundle:
    bundle = st.session_state.bundle
    
    # Retrieve components from the bundle
    leaderboard = bundle.get('leaderboard')
    feature_importance = bundle.get('feature_importance')
    metrics = bundle.get('metrics')
    automl = bundle.get('best_model')
    test_data = bundle.get('data')['test']
    feature_cols = bundle.get('data')['feature_cols']
    train_target_col = bundle.get('data')['target_col']
    
    # Retrieve the engine instance for data access
    engine_instance = bundle.get('class_instance')
    if engine_instance:
        df_processed = engine_instance.df_processed
        processed_date_col = engine_instance.date_col
        processed_target_col = engine_instance.target_col
    else:
        st.error("Model bundle is incomplete. Please retrain.")
        st.stop()

    st.title(f"Research Lab: {processed_target_col} Prediction")
    
    # Four main tabs for the analytics workflow
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Deep Data Scan", 
        "⚔️ Model Arena", 
        "🔍 Diagnostics", 
        "🔮 Future Forecast"
    ])

    # --- Tab 1: Deep Data Scan (EDA) ---
    with tab1:
        st.header("Deep Data Scan")
        st.markdown("Automatic exploration of time-series properties.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Time Series Decomposition")
            try:
                # Use processed data which is already aggregated by date
                ts_data = df_processed.set_index(processed_date_col)[processed_target_col]
                
                if len(ts_data) > 14:
                    # Decompose into Trend, Seasonality, and Residuals (Period=7 for weekly seasonality)
                    decomp = seasonal_decompose(ts_data, model='additive', period=7)
                    
                    fig_trend = px.line(x=ts_data.index, y=decomp.trend, title="Trend Component")
                    fig_trend.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_trend, width='stretch')
                    
                    fig_season = px.line(x=ts_data.index, y=decomp.seasonal, title="Seasonal Component")
                    fig_season.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_season, width='stretch')
                else:
                    st.warning("Not enough data for decomposition (need > 14 points).")
            except Exception as e:
                st.error(f"Decomposition failed: {e}")

        with col2:
            st.subheader("Correlation Heatmap")
            # Analyze relationships between numerical features
            numeric_df = df_processed.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Feature Correlation")
            fig_corr.update_layout(template="plotly_dark")
            st.plotly_chart(fig_corr, width='stretch')

        st.subheader("Autocorrelation Analysis")
        try:
            ts_values = df_processed[processed_target_col].values
            # ACF (AutoCorrelation Function) - Direct correlation with lags
            acf_values = sm.tsa.acf(ts_values, nlags=20)
            fig_acf = px.bar(x=list(range(len(acf_values))), y=acf_values, title="Autocorrelation (ACF)")
            fig_acf.update_layout(template="plotly_dark", xaxis_title="Lag", yaxis_title="ACF")
            
            # PACF (Partial AutoCorrelation Function) - Direct correlation removing intermediate lags
            pacf_values = sm.tsa.pacf(ts_values, nlags=20)
            fig_pacf = px.bar(x=list(range(len(pacf_values))), y=pacf_values, title="Partial Autocorrelation (PACF)")
            fig_pacf.update_layout(template="plotly_dark", xaxis_title="Lag", yaxis_title="PACF")
            
            c1, c2 = st.columns(2)
            c1.plotly_chart(fig_acf, width='stretch')
            c2.plotly_chart(fig_pacf, width='stretch')
        except Exception as e:
            st.warning(f"ACF/PACF analysis failed: {e}")

    # --- Tab 2: Model Arena ---
    with tab2:
        st.header("Model Arena")
        st.markdown("Comparison of different machine learning models.")
        
        winner = "Unknown"
        if leaderboard is not None and not leaderboard.empty:
            winner = leaderboard.iloc[0]['Model_Type']
        
        st.info(f"🏆 Champion Model: **{winner}**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Leaderboard")
            if leaderboard is not None:
                st.dataframe(leaderboard.style.highlight_min(subset=['RMSE'], color='#00ADB5'), width='stretch')
            else:
                st.write("No leaderboard data.")
            
        with col2:
            st.subheader("Global Feature Importance")
            if feature_importance:
                fi_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
                fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Viridis', title="Top Drivers of Sales")
                fig_fi.update_layout(template="plotly_dark")
                st.plotly_chart(fig_fi, width='stretch')
            else:
                st.info("Feature importance not available for this model type.")

        # Training History
        st.subheader("Training History")
        if os.path.exists("flaml.log"):
            try:
                # Parse FLAML log to show improvement over time
                log_data = []
                with open("flaml.log", "r") as f:
                    import json
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if 'curr_loss' in entry and 'wall_clock_time' in entry:
                                log_data.append(entry)
                        except:
                            pass
                
                if log_data:
                    hist_df = pd.DataFrame(log_data)
                    fig_hist = px.line(hist_df, x='wall_clock_time', y='curr_loss', title="Loss Optimization over Time")
                    fig_hist.update_layout(template="plotly_dark", xaxis_title="Time (s)", yaxis_title="RMSE")
                    st.plotly_chart(fig_hist, width='stretch')
                else:
                    st.write("Log file empty or unreadable.")
            except Exception as e:
                st.write(f"Could not read training history: {e}")

    # --- Tab 3: Diagnostics ---
    with tab3:
        st.header("Diagnostics")
        st.markdown("Detailed error analysis on the hold-out test set.")
        
        X_test, y_test = test_data
        y_pred = automl.predict(X_test)
        residuals = y_test.values - y_pred
        
        st.subheader("Actual vs Predicted (Test Set)")
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(y=y_test.values, name='Actual', mode='lines', line=dict(color='#EEEEEE')))
        fig_avp.add_trace(go.Scatter(y=y_pred, name='Predicted', mode='lines', line=dict(color='#00ADB5', dash='dot')))
        fig_avp.update_layout(template="plotly_dark", title="Model Performance on Unseen Data")
        st.plotly_chart(fig_avp, width='stretch')
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Residual Histogram")
            # Check for Gaussian distribution of errors
            fig_hist = px.histogram(residuals, nbins=30, title="Error Distribution (Should be Gaussian)")
            fig_hist.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_hist, width='stretch')
            
        with col2:
            st.subheader("Residuals vs Predicted")
            # Check for Homoscedasticity (Equal variance across predictions)
            fig_res = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted', 'y': 'Residuals'}, title="Homoscedasticity Check")
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            fig_res.update_layout(template="plotly_dark")
            st.plotly_chart(fig_res, width='stretch')

    # --- Tab 4: Future Forecast ---
    with tab4:
        st.header("Future Forecast")
        st.markdown("Generate predictions for future dates using recursive forecasting.")
        
        days_to_forecast = st.slider("Days to Forecast", 1, 30, 7)
        
        if st.button("Generate Forecast"):
            with st.spinner("Calculating future trajectories..."):
                last_date = df_processed[processed_date_col].max()
                
                # Setup for recursive prediction
                future_preds = []
                future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, days_to_forecast + 1)]
                
                # Get the full history of the target to compute lags
                history_target = df_processed[processed_target_col].tolist()
                
                # Get last row to pick up exogenous features (assuming constant for future)
                last_row = df_processed.iloc[-1].to_dict()
                
                for date in future_dates:
                    # 1. Date Features
                    row = last_row.copy()
                    row['DayOfWeek'] = date.dayofweek
                    row['Month'] = date.month
                    row['Day'] = date.day
                    row['Year'] = date.year
                    
                    # 2. Lag Features (Recursive updates)
                    # Lag_1 = T-1
                    row['Lag_1'] = history_target[-1]
                    # Lag_7 = T-7
                    row['Lag_7'] = history_target[-7] if len(history_target) >= 7 else history_target[0]
                    # Rolling_Mean_7
                    row['Rolling_Mean_7'] = np.mean(history_target[-7:])
                    
                    # Construct DataFrame for prediction
                    X_future = pd.DataFrame([row])[feature_cols]
                    
                    # Predict
                    pred = automl.predict(X_future)[0]
                    
                    future_preds.append(pred)
                    history_target.append(pred) # Append forecast to history for next iteration
                
                # Display Results
                forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': future_preds})
                
                fig_cast = px.line(forecast_df, x='Date', y='Forecast', markers=True, title=f"Forecast for next {days_to_forecast} Days")
                fig_cast.update_layout(template="plotly_dark")
                
                st.plotly_chart(fig_cast, width='stretch')
                
                st.write("Forecast Values:")
                st.dataframe(forecast_df)

else:
    # Landing Page
    st.info("👋 Welcome to the Research Lab. Please Upload Data or use Demo Data to start.")
    if os.path.exists("supermarket_sales.csv"):
        st.markdown("Found `supermarket_sales.csv`. Select 'Use Demo Data' in sidebar.")
