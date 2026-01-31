import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder

class SalesDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    A modular preprocessor class for the Supermarket Sales Prediction project.
    It handles data cleaning (dropping unused columns via remainder='drop'),
    feature engineering (Target Encoding for high-cardinality, One-Hot Encoding for others),
    and numerical scaling.
    """
    def __init__(self, numerical_features=None, categorical_features_ohe=None, categorical_features_te=None):
        """
        Initialize the preprocessor with feature lists.
        
        Parameters:
        numerical_features (list): List of numerical column names to scale.
        categorical_features_ohe (list): List of categorical column names to One-Hot Encode.
        categorical_features_te (list): List of categorical column names to Target Encode.
        """
        # Default configuration based on the dataset analysis
        self.numerical_features = numerical_features if numerical_features is not None else [
            'Unit price', 'Quantity', 'Tax 5%', 'cogs', 'gross income', 'Rating'
        ]
        
        self.categorical_features_ohe = categorical_features_ohe if categorical_features_ohe is not None else [
            'Branch', 'City', 'Customer type', 'Gender', 'Payment'
        ]
        
        self.categorical_features_te = categorical_features_te if categorical_features_te is not None else [
            'Product line'
        ]
        
        self.preprocessor_ = None

    def fit(self, X, y=None):
        """
        Fit the preprocessor to the data.
        
        Parameters:
        X (pd.DataFrame): The training input samples.
        y (pd.Series): The target values (required for TargetEncoder).
        """
        # Define transformers
        transformers = [
            ('num', StandardScaler(), self.numerical_features),
            ('cat_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features_ohe),
            ('cat_te', TargetEncoder(target_type='continuous'), self.categorical_features_te)
        ]

        # Create ColumnTransformer
        # remainder='drop' ensures columns like 'Invoice ID', 'Date', 'Time' are removed
        self.preprocessor_ = ColumnTransformer(
            transformers=transformers, 
            remainder='drop',
            verbose_feature_names_out=False
        )
        
        self.preprocessor_.fit(X, y)
        return self

    def transform(self, X):
        """
        Transform the data using the fitted preprocessor.
        
        Parameters:
        X (pd.DataFrame): The input samples to transform.
        """
        if self.preprocessor_ is None:
            raise RuntimeError("The preprocessor has not been fitted yet.")
        
        return self.preprocessor_.transform(X)

    def get_feature_names_out(self):
        """
        Get output feature names for transformation.
        """
        if self.preprocessor_ is None:
            raise RuntimeError("The preprocessor has not been fitted yet.")
            
        return self.preprocessor_.get_feature_names_out()
