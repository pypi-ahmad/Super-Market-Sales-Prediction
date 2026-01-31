"""
Universal Forecasting Engine powered by Microsoft FLAML.
Handles data preprocessing, feature engineering, and AutoML model selection.
"""

import pandas as pd
import numpy as np
import logging
import joblib
import warnings
from flaml import AutoML
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress warnings for cleaner logs
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ForecastingEngine:
    """
    A comprehensive engine for time-series forecasting using AutoML.
    
    Attributes:
        time_budget (int): Time in seconds for the AutoML search.
        automl (flaml.AutoML): The trained AutoML object.
        best_model_name (str): The name of the winning algorithm.
        metrics (dict): Performance metrics of the best model.
        feature_cols (list): List of feature names used for training.
        date_col (str): Name of the date column.
        target_col (str): Name of the target column.
        df_processed (pd.DataFrame): The dataframe after preprocessing.
        test_data (tuple): (X_test, y_test) split for final evaluation.
        leaderboard (pd.DataFrame): Comparison of all tried models.
        feature_importance (dict): Feature importance scores.
    """

    def __init__(self, time_budget=30):
        """
        Initialize the forecasting engine.

        Args:
            time_budget (int): The maximum time (in seconds) allowed for FLAML to search for models.
        """
        self.time_budget = time_budget
        self.automl = None
        self.best_model_name = None
        self.metrics = {}
        self.feature_cols = []
        self.date_col = None
        self.target_col = None
        self.df_processed = None
        self.test_data = None 
        self.leaderboard = None
        self.feature_importance = None
        self.training_history = None

    def preprocess(self, df, date_col, target_col):
        """
        Prepares the raw data for time-series forecasting.
        
        Performs:
        1. Date parsing and sorting.
        2. Aggregation (if multiple records exist for the same date).
        3. Handling missing values in the target.
        4. Feature Engineering: Date parts (Month, Day) and Lags (T-1, T-7).

        Args:
            df (pd.DataFrame): Raw input dataframe.
            date_col (str): Name of the column containing date strings.
            target_col (str): Name of the numeric target column to predict.

        Returns:
            pd.DataFrame: The processed dataframe ready for training.
        """
        logger.info("Starting preprocessing...")
        self.date_col = date_col
        self.target_col = target_col
        
        # Copy to avoid modifying original
        data = df.copy()
        
        # Parse Dates
        try:
            data[date_col] = pd.to_datetime(data[date_col])
        except Exception as e:
            logger.error(f"Date parsing failed: {e}")
            raise
            
        data = data.sort_values(by=date_col)
        
        # Aggregate if multiple entries per date (e.g., multiple sales in one day)
        if data.groupby(date_col).size().max() > 1:
             numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
             if target_col in numeric_cols:
                data = data.groupby(date_col)[numeric_cols].sum().reset_index()
             else:
                # Fallback if target is not numeric (should not happen for regression)
                pass

        # Handle Missing Target (Forward fill then backward fill)
        if data[target_col].isnull().any():
            data[target_col] = data[target_col].ffill().bfill()
        
        # --- Feature Engineering ---
        
        # 1. Date Parts (Captures Seasonality)
        data['DayOfWeek'] = data[date_col].dt.dayofweek
        data['Month'] = data[date_col].dt.month
        data['Day'] = data[date_col].dt.day
        data['Year'] = data[date_col].dt.year
        
        # 2. Lag Features (Captures Autocorrelation)
        data['Lag_1'] = data[target_col].shift(1)  # Yesterday's sales
        data['Lag_7'] = data[target_col].shift(7)  # Last week's sales
        data['Rolling_Mean_7'] = data[target_col].shift(1).rolling(window=7).mean()  # Weekly trend
        
        # Drop NaNs created by lags (we lose the first 7 days)
        data = data.dropna().reset_index(drop=True)
        
        self.df_processed = data
        self.feature_cols = [c for c in data.columns if c not in [date_col, target_col]]
        
        return self.df_processed

    def train(self):
        """
        Trains the FLAML AutoML model.

        Splits data into Train/Test, configures the search space, and finds the best model.
        Also extracts the leaderboard and feature importance after training.
        """
        if self.df_processed is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")

        logger.info(f"Splitting data for training (Target: {self.target_col})...")
        
        X = self.df_processed[self.feature_cols]
        y = self.df_processed[self.target_col]
        
        # Time-based split (simple 80/20) to respect temporal order
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.test_data = (X_test, y_test) # Save for evaluation

        logger.info(f"Initializing FLAML AutoML (Time Budget: {self.time_budget}s)...")
        self.automl = AutoML()
        
        settings = {
            "time_budget": self.time_budget,  # total running time in seconds
            "metric": 'rmse', 
            "task": 'regression',
            "estimator_list": ['lgbm', 'xgboost', 'rf', 'extra_tree'],
            "log_file_name": "flaml.log",
            "verbose": 0
        }
        
        self.automl.fit(X_train=X_train, y_train=y_train, **settings)
        
        logger.info(f"Best estimator: {self.automl.best_estimator}")
        logger.info(f"Best loss: {self.automl.best_loss}")
        self.best_model_name = self.automl.best_estimator

        # Extract metadata for visualization
        self._extract_results()

    def _extract_results(self):
        """
        Internal helper to extract leaderboard, feature importance, and training history from FLAML.
        """
        # 1. Leaderboard (Model Comparison)
        results = []
        if hasattr(self.automl, 'best_loss_per_estimator'):
            for model_name, loss in self.automl.best_loss_per_estimator.items():
                results.append({
                    'Model_Type': model_name,
                    'RMSE': loss, 
                    'MAE': np.nan, # FLAML optimizes one metric, simplified here
                    'Training_Time': self.automl.best_config_per_estimator[model_name].get('time_total_s', np.nan) if hasattr(self.automl, 'best_config_per_estimator') else np.nan
                })
        self.leaderboard = pd.DataFrame(results).sort_values(by='RMSE')

        # 2. Feature Importance (Explainability)
        try:
            # Handle different model types (Tree-based usually have feature_importances_)
            if hasattr(self.automl.model, 'estimator') and hasattr(self.automl.model.estimator, 'feature_importances_'):
                fi = self.automl.model.estimator.feature_importances_
                self.feature_importance = dict(zip(self.feature_cols, fi))
            elif hasattr(self.automl.model, 'feature_importances_'):
                fi = self.automl.model.feature_importances_
                self.feature_importance = dict(zip(self.feature_cols, fi))
            else:
                self.feature_importance = {}
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            self.feature_importance = {}

    def get_leaderboard(self):
        """
        Calculates final metrics on the held-out test set.

        Returns:
            pd.DataFrame: A dataframe containing RMSE, MAE, R2, and the Winner Model name.
        """
        if self.automl is None:
            return pd.DataFrame()

        # Evaluate on test set
        X_test, y_test = self.test_data
        y_pred = self.automl.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.metrics = {
            "Winner Model": [self.best_model_name],
            "RMSE": [rmse],
            "MAE": [mae],
            "R2": [r2],
            "Best Config": [str(self.automl.best_config)]
        }
        
        return pd.DataFrame(self.metrics)

    def save(self, filename='automl_bundle.pkl'):
        """
        Saves the entire engine state and results to a pickle file.
        
        Args:
            filename (str): The path to save the .pkl file.
        """
        logger.info(f"Saving comprehensive bundle to {filename}...")
        
        bundle = {
            'best_model': self.automl,
            'leaderboard': self.leaderboard,
            'feature_importance': self.feature_importance,
            'data': {
                'train': None, # We don't save full train to save space
                'test': self.test_data,
                'feature_cols': self.feature_cols,
                'target_col': self.target_col
            },
            'metrics': self.metrics,
            'class_instance': self # Save the full engine state
        }
        
        joblib.dump(bundle, filename)
        logger.info("Save complete.")

if __name__ == "__main__":
    # Test Execution when run directly
    try:
        import os
        if os.path.exists("supermarket_sales.csv"):
            df = pd.read_csv("supermarket_sales.csv")
            engine = ForecastingEngine(time_budget=10)
            engine.preprocess(df, date_col='Date', target_col='Total')
            engine.train()
            print(engine.get_leaderboard())
            engine.save()
            print("✅ FLAML Pipeline Finished Successfully!")
        else:
            print("Skipping test run: supermarket_sales.csv not found.")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")

