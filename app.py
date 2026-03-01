import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import pickle
import hashlib
import logging
import matplotlib.pyplot as plt

# Try importing SHAP, handle if missing
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- Configuration ---
st.set_page_config(
    page_title="Supermarket Sales AI Command Center",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_FILE = 'automl_model.pkl'

logger = logging.getLogger(__name__)


def _model_hash_file():
    return f"{MODEL_FILE}.sha256"

# --- Helper Functions ---
@st.cache_data
def load_data(file):
    file_name = str(getattr(file, 'name', '')).lower()
    if file_name.endswith('.csv'):
        return pd.read_csv(file)
    if file_name.endswith('.xlsx'):
        return pd.read_excel(file)
    raise ValueError("Unsupported file type. Please upload a CSV or XLSX file.")


def _compute_file_sha256(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_column_metadata(df):
    """Generate metadata for dynamic inputs."""
    metadata = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            metadata[col] = {
                'type': 'categorical',
                'options': sorted(df[col].unique().astype(str).tolist())
            }
        elif pd.api.types.is_numeric_dtype(df[col]):
            metadata[col] = {
                'type': 'numeric',
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean())
            }
    return metadata

def save_model(automl, features, metadata, metrics, config):
    artifacts = {
        'model': automl,
        'features': features,
        'column_metadata': metadata,
        'metrics': metrics,
        'config': config
    }
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(artifacts, f)
    with open(_model_hash_file(), 'w', encoding='utf-8') as hash_file:
        hash_file.write(_compute_file_sha256(MODEL_FILE))
    return artifacts

def load_pretrained_model():
    if os.path.exists(MODEL_FILE):
        try:
            model_hash_file = _model_hash_file()
            if os.path.exists(model_hash_file):
                with open(model_hash_file, 'r', encoding='utf-8') as hash_file:
                    expected_hash = hash_file.read().strip()
                if expected_hash != _compute_file_sha256(MODEL_FILE):
                    logger.error("Model integrity check failed.")
                    return None

            with open(MODEL_FILE, 'rb') as f:
                artifacts = pickle.load(f)

            required_keys = {'model', 'features', 'column_metadata', 'metrics', 'config'}
            if not isinstance(artifacts, dict) or not required_keys.issubset(artifacts.keys()):
                logger.error("Model artifact structure is invalid.")
                return None

            return artifacts
        except Exception as error:
            logger.error(f"Failed to load pre-trained model: {error}")
            return None
    return None

# --- Main App ---
def main():
    st.title("🛒 Supermarket Sales AI Command Center")
    st.markdown("### Full-Stack AutoML & Analytics Platform")

    # --- Sidebar Configuration ---
    st.sidebar.header("🔧 Configuration")
    
    data_source = st.sidebar.radio(
        "Data Source",
        ["Use Pre-trained Model", "Train on New Data"],
        help="Choose between using the existing model or training a new one."
    )

    # State Management for Data and Model
    if 'active_model' not in st.session_state:
        st.session_state.active_model = None
    if 'active_meta' not in st.session_state:
        st.session_state.active_meta = None
    if 'data_df' not in st.session_state:
        st.session_state.data_df = None
    if 'uploaded_filename' not in st.session_state:
        st.session_state.uploaded_filename = None

    # Logic based on Data Source
    if data_source == "Use Pre-trained Model":
        artifacts = load_pretrained_model()
        if artifacts:
            st.session_state.active_model = artifacts['model']
            st.session_state.active_meta = artifacts
            st.sidebar.success(f"Loaded Model: {artifacts['config'].get('app_title', 'Unknown')}")
            
            # Try to load default data for EDA/SHAP if available
            if os.path.exists("supermarket_sales.csv"):
                # Load lazily or check if already loaded
                if st.session_state.data_df is None:
                    st.session_state.data_df = pd.read_csv("supermarket_sales.csv")
        else:
            st.sidebar.error(f"No pre-trained model found at {MODEL_FILE}")
            
    else: # Train on New Data
        uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV/Excel)", type=['csv', 'xlsx'])
        if uploaded_file:
            try:
                if st.session_state.uploaded_filename != uploaded_file.name:
                    df = load_data(uploaded_file)
                    st.session_state.data_df = df
                    st.session_state.uploaded_filename = uploaded_file.name
                st.sidebar.success(f"Uploaded: {uploaded_file.name} ({len(st.session_state.data_df)} rows)")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Data Explorer", 
        "⚙️ AutoML Training Lab", 
        "🛒 Sales Simulator", 
        "🧠 Model Explainability"
    ])

    # --- Tab 1: Data Explorer ---
    with tab1:
        st.header("🔍 Data Explorer")
        df = st.session_state.data_df
        
        if df is not None:
            # 1. Stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing Values", df.isna().sum().sum())
            
            with st.expander("View Raw Data", expanded=False):
                st.dataframe(df.head(100))
                
            # 2. Visuals
            st.subheader("Visual Analytics")
            
            # Select target for distribution
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                target_col = st.selectbox("Select Column to Analyze Distribution", numeric_cols, index=0)
                
                c1, c2 = st.columns(2)
                with c1:
                    fig_hist = px.histogram(df, x=target_col, title=f"Distribution of {target_col}", nbins=30)
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with c2:
                    # Correlation Heatmap
                    if len(numeric_cols) > 1:
                        corr = df[numeric_cols].corr()
                        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Heatmap")
                        st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.info("Not enough numeric columns for correlation.")
            else:
                st.warning("No numeric columns found for visualization.")

            # Categorical bars
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            if cat_cols:
                st.subheader("Categorical Distributions")
                selected_cat = st.selectbox("Select Categorical Column", cat_cols)
                fig_bar = px.bar(df[selected_cat].value_counts().reset_index(), x=selected_cat, y='count', title=f"Count of {selected_cat}")
                st.plotly_chart(fig_bar, use_container_width=True)
                
        else:
            st.info("Please upload data or load a model with default data to view explorer.")

    # --- Tab 2: AutoML Training Lab ---
    with tab2:
        st.header("⚙️ AutoML Training Lab")
        
        if data_source == "Train on New Data" and st.session_state.data_df is not None:
            df = st.session_state.data_df
            columns = df.columns.tolist()
            
            c1, c2, c3 = st.columns(3)
            target_col = c1.selectbox("Select Target Column", columns, index=len(columns)-1 if 'Total' not in columns else columns.index('Total'))
            time_budget = c2.slider("Time Budget (seconds)", 30, 600, 60)
            use_gpu = c3.checkbox("Use GPU", value=False)
            
            if st.button("🚀 Start Training"):
                st.info(f"Training AutoML model to predict '{target_col}'...")
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Preprocessing
                status_text.text("Preprocessing Data...")
                progress_bar.progress(10)
                
                if target_col not in df.columns:
                    st.error(f"Target column '{target_col}' not found.")
                    st.stop()

                training_df = df.dropna(subset=[target_col]).copy()
                if training_df.empty:
                    st.error("No rows available after dropping missing target values.")
                    st.stop()

                leakage_cols = [
                    'Invoice ID',
                    'Tax 5%',
                    'cogs',
                    'gross income',
                    'gross margin percentage',
                    'Date',
                    'Time'
                ]
                drop_cols = [c for c in leakage_cols if c in training_df.columns and c != target_col]

                X = training_df.drop(columns=[target_col] + drop_cols)
                y = training_df[target_col]

                valid_rows = X.notna().all(axis=1) & y.notna()
                X = X.loc[valid_rows]
                y = y.loc[valid_rows]

                if len(X) < 2:
                    st.error("Not enough valid rows to train after removing missing values.")
                    st.stop()
                
                # Metadata
                metadata = get_column_metadata(X)
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                status_text.text(f"Running FLAML (Budget: {time_budget}s)...")
                progress_bar.progress(30)
                
                # AutoML
                automl = AutoML()
                settings = {
                    "time_budget": time_budget,
                    "metric": 'r2',
                    "task": 'regression',
                    "verbose": 0,
                    "seed": 42,
                    "use_gpu": use_gpu
                }
                
                automl.fit(X_train=X_train, y_train=y_train, **settings)
                
                progress_bar.progress(90)
                status_text.text("Evaluating Model...")
                
                # Metrics
                y_pred = automl.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                
                metrics = {'r2': r2, 'mae': mae, 'mse': mse}
                
                # Save to session
                config = {'target': target_col, 'app_title': "Custom Trained Model"}
                
                # Save to disk
                artifacts = save_model(automl, X.columns.tolist(), metadata, metrics, config)
                st.session_state.active_model = artifacts['model']
                st.session_state.active_meta = artifacts
                
                progress_bar.progress(100)
                status_text.text("Training Complete!")
                
                st.success("Model Trained Successfully!")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("R² Score", f"{r2:.4f}")
                c2.metric("MAE", f"{mae:.4f}")
                c3.metric("RMSE", f"{np.sqrt(mse):.4f}")
                
        elif data_source == "Use Pre-trained Model":
            st.warning("Switch to 'Train on New Data' in the sidebar to use this lab.")
        else:
            st.info("Please upload a dataset to start training.")

    # --- Tab 3: Sales Simulator ---
    with tab3:
        st.header("🛒 Sales Simulator")
        
        if st.session_state.active_model and st.session_state.active_meta:
            artifacts = st.session_state.active_meta
            model = st.session_state.active_model
            features = artifacts['features']
            metadata = artifacts['column_metadata']
            target_name = artifacts['config'].get('target', 'Target')
            
            st.markdown(f"Predicting: **{target_name}**")
            
            # Dynamic Form
            input_data = {}
            with st.form("simulation_form"):
                cols = st.columns(3) # Grid layout
                for i, col_name in enumerate(features):
                    col_info = metadata.get(col_name, {})
                    col_type = col_info.get('type', 'unknown')
                    
                    with cols[i % 3]:
                        if col_type == 'categorical':
                            options = col_info.get('options', [])
                            input_data[col_name] = st.selectbox(f"{col_name}", options)
                        elif col_type == 'numeric':
                            min_val = col_info.get('min', 0.0)
                            max_val = col_info.get('max', 10000.0)
                            mean_val = col_info.get('mean', 0.0)
                            if not np.isfinite(min_val):
                                min_val = 0.0
                            if not np.isfinite(max_val):
                                max_val = min_val + 1.0
                            if min_val > max_val:
                                min_val, max_val = max_val, min_val
                            default_val = mean_val
                            if not np.isfinite(default_val):
                                default_val = min_val
                            default_val = min(max(default_val, min_val), max_val)
                            input_data[col_name] = st.number_input(
                                f"{col_name}",
                                min_value=float(min_val),
                                max_value=float(max_val),
                                value=float(default_val)
                            )
                        else:
                            input_data[col_name] = st.text_input(f"{col_name}")
                
                predict_btn = st.form_submit_button("Predict Sales")
            
            if predict_btn:
                input_df = pd.DataFrame([input_data])
                
                try:
                    prediction = model.predict(input_df)[0]
                    st.metric("Predicted Value", f"{prediction:,.2f}")
                    
                    # Store for explanation
                    st.session_state.last_prediction_input = input_df
                    st.session_state.last_prediction_value = prediction
                    
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    
        else:
            st.warning("No active model. Please load or train a model.")

    # --- Tab 4: Model Explainability ---
    with tab4:
        st.header("🧠 Model Explainability")
        
        if not SHAP_AVAILABLE:
            st.error("SHAP library not found. Please install it to use this feature.")
        elif st.session_state.active_model:
            model = st.session_state.active_model
            
            # We need background data for SHAP
            if st.session_state.data_df is not None:
                df = st.session_state.data_df
                features = st.session_state.active_meta['features']
                
                # Check if features exist in df
                if all(f in df.columns for f in features):
                    X_background = df[features].head(100) # Use subset for speed
                    
                    # Initialize Explainer
                    try:
                        # Warning: SHAP with general function can be slow.
                        # We limit background to 50 samples for speed.
                        X_background_small = X_background.iloc[:50]
                        
                        # Handle Categorical Encoding for SHAP
                        # SHAP needs numeric data to calculate perturbations/variance.
                        cat_cols = X_background_small.select_dtypes(include=['object']).columns
                        
                        if len(cat_cols) > 0:
                            # 1. Create Mappings
                            encoders = {}
                            X_bg_encoded = X_background_small.copy()
                            
                            for col in cat_cols:
                                unique_vals = X_background_small[col].dropna().unique().tolist()
                                val_to_int = {val: i for i, val in enumerate(unique_vals)}
                                int_to_val = {i: val for val, i in val_to_int.items()}
                                fallback_value = unique_vals[0] if unique_vals else ""
                                encoders[col] = {
                                    'val_to_int': val_to_int,
                                    'int_to_val': int_to_val,
                                    'fallback': fallback_value
                                }
                                X_bg_encoded[col] = X_background_small[col].map(val_to_int).fillna(-1)

                            # 2. Create Wrapper
                            def predict_wrapper(X_numeric):
                                # X_numeric comes from SHAP as numpy or DataFrame
                                if isinstance(X_numeric, np.ndarray):
                                    X_temp = pd.DataFrame(X_numeric, columns=X_background_small.columns)
                                else:
                                    X_temp = X_numeric.copy()
                                
                                # Decode back to strings
                                for col, encoder in encoders.items():
                                    X_temp[col] = X_temp[col].round().map(encoder['int_to_val']).fillna(encoder['fallback'])
                                
                                return model.predict(X_temp)
                            
                            # 3. Use Encoded Data with Wrapper
                            explainer = shap.Explainer(predict_wrapper, X_bg_encoded)
                            
                            st.subheader("Global Feature Importance")
                            with st.spinner("Calculating Global Importance (Beeswarm)..."):
                                shap_values = explainer(X_bg_encoded)
                                
                                fig, ax = plt.subplots()
                                shap.plots.beeswarm(shap_values, show=False)
                                st.pyplot(fig)
                                plt.close()
                                
                            # Local Explanation
                            if 'last_prediction_input' in st.session_state:
                                st.subheader("Local Explanation (Waterfall)")
                                st.markdown("Explaining the most recent prediction from Tab 3.")
                                
                                input_row = st.session_state.last_prediction_input.copy()
                                
                                # Encode input_row using same encoders
                                for col, encoder in encoders.items():
                                    input_row[col] = input_row[col].map(encoder['val_to_int']).fillna(-1)

                                # Calculate SHAP for single instance
                                shap_single = explainer(input_row)
                                
                                fig2, ax2 = plt.subplots()
                                shap.plots.waterfall(shap_single[0], show=False)
                                st.pyplot(fig2)
                                plt.close()
                            else:
                                st.info("Make a prediction in the 'Sales Simulator' tab to see local explanation.")

                        else:
                            # Numeric Only - simpler path
                            explainer = shap.Explainer(model.predict, X_background_small)
                            
                            st.subheader("Global Feature Importance")
                            with st.spinner("Calculating Global Importance (Beeswarm)..."):
                                shap_values = explainer(X_background_small)
                                
                                fig, ax = plt.subplots()
                                shap.plots.beeswarm(shap_values, show=False)
                                st.pyplot(fig)
                                plt.close()

                            # Local Explanation
                            if 'last_prediction_input' in st.session_state:
                                st.subheader("Local Explanation (Waterfall)")
                                st.markdown("Explaining the most recent prediction from Tab 3.")
                                
                                input_row = st.session_state.last_prediction_input
                                shap_single = explainer(input_row)
                                
                                fig2, ax2 = plt.subplots()
                                shap.plots.waterfall(shap_single[0], show=False)
                                st.pyplot(fig2)
                                plt.close()
                            else:
                                st.info("Make a prediction in the 'Sales Simulator' tab to see local explanation.")
                            
                    except Exception as e:
                        st.error(f"Could not generate SHAP plots: {e}")
                        st.warning("Note: SHAP explanation with AutoML pipelines can be complex due to internal preprocessing.")
                else:
                    st.warning("Loaded data does not match model features. Cannot run SHAP.")
            else:
                st.warning("No background data available for SHAP. Please upload the dataset used for training.")
        else:
             st.warning("No active model.")

if __name__ == "__main__":
    main()
