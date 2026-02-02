import pandas as pd
from flaml import AutoML
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import logging
import os

# Configuration
DATA_SOURCE = "supermarket_sales.csv"
TARGET = "Total"
APP_TITLE = "Supermarket Sales AI 🛒"
TIME_BUDGET = 90
MODEL_FILE = 'automl_model.pkl'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print(f"--- {APP_TITLE} ---")
    logging.info(f"Starting AutoML Training Pipeline...")
    
    # 1. Load Data
    if not os.path.exists(DATA_SOURCE):
        logging.error(f"Data file '{DATA_SOURCE}' not found.")
        return

    try:
        df = pd.read_csv(DATA_SOURCE)
        logging.info(f"Data loaded successfully from {DATA_SOURCE}. Shape: {df.shape}")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    # 2. Preprocessing
    if TARGET not in df.columns:
        logging.error(f"Target column '{TARGET}' not found in dataset.")
        return

    # Basic cleanup: Drop irrelevant ID columns
    if 'Invoice ID' in df.columns:
        df = df.drop(columns=['Invoice ID'])

    # Separate features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Capture column metadata for the App
    column_metadata = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            # Handle categorical
            column_metadata[col] = {
                'type': 'categorical',
                'options': sorted(X[col].unique().astype(str).tolist())
            }
        elif pd.api.types.is_numeric_dtype(X[col]):
            # Handle numeric
            column_metadata[col] = {
                'type': 'numeric',
                'min': float(X[col].min()),
                'max': float(X[col].max()),
                'mean': float(X[col].mean())
            }

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Data split into train ({len(X_train)}) and test ({len(X_test)}) sets.")

    # 3. Initialize and Run AutoML
    automl = AutoML()
    settings = {
        "time_budget": TIME_BUDGET,
        "metric": 'r2',
        "task": 'regression',
        "log_file_name": 'flaml.log',
        "seed": 42,
        "verbose": 1
    }
    
    logging.info(f"Starting AutoML training with time budget {TIME_BUDGET}s...")
    automl.fit(X_train=X_train, y_train=y_train, **settings)
    
    # 4. Evaluation
    logging.info(f"Best ML learner: {automl.best_estimator}")
    logging.info(f"Best hyperparameter config: {automl.best_config}")
    
    y_pred = automl.predict(X_test)
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred)
    }
    
    logging.info(f"Test Set Metrics: {metrics}")

    # 5. Save Artifacts Bundle
    artifacts = {
        'model': automl,
        'features': X.columns.tolist(),
        'column_metadata': column_metadata,
        'metrics': metrics,
        'config': {
            'target': TARGET,
            'app_title': APP_TITLE
        }
    }

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(artifacts, f)
    logging.info(f"Model and artifacts saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()
