from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# Load the trained models
with open('saved_models/xgb_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

with open('saved_models/svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

with open('saved_models/ridge_model.pkl', 'rb') as file:
    ridge_model = pickle.load(file)

with open('saved_models/rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('saved_models/mlp_model.pkl', 'rb') as file:
    mlp_model = pickle.load(file)

with open('saved_models/lgbm_model.pkl', 'rb') as file:
    lgbm_model = pickle.load(file)

with open('saved_models/lasso_model.pkl', 'rb') as file:
    lasso_model = pickle.load(file)

with open('saved_models/knn_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)

with open('saved_models/catboost_model.pkl', 'rb') as file:
    catboost_model = pickle.load(file)

# Load the StandardScaler used during training
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the original feature names
feature_names = ['Branch', 'City', 'Customer type', 'Gender', 'Product line',
       'Unit price', 'Quantity', 'Tax 5%', 'cogs', 'gross income', 'Rating']

# Define the encoded feature names
encoded_feature_names = [
    'Unit price', 'Quantity', 'Tax 5%', 'cogs', 'gross income', 'Rating',
    'Branch_A', 'Branch_B', 'Branch_C', 'City_Mandalay', 'City_Naypyitaw', 'City_Yangon',
    'Customer type_Member', 'Customer type_Normal', 'Gender_Female', 'Gender_Male',
    'Product line_Electronic accessories', 'Product line_Fashion accessories',
    'Product line_Food and beverages', 'Product line_Health and beauty',
    'Product line_Home and lifestyle', 'Product line_Sports and travel',
    'Payment_Cash', 'Payment_Credit card', 'Payment_Ewallet'
]

# Render the HTML form with input fields for each feature
@app.route('/')
def index():
    return render_template('index.html', feature_names=feature_names)

# Preprocess input data and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input values from the form
        input_values = [request.form.get(feat) for feat in feature_names]

        # Create a DataFrame from input values
        input_df = pd.DataFrame([input_values], columns=feature_names)

        # Convert input to the encoded format
        input_encoded = pd.get_dummies(input_df, columns=['Branch', 'City', 'Customer type', 'Gender', 'Product line'], drop_first=False)
        input_encoded = input_encoded.reindex(columns=encoded_feature_names, fill_value=0)

        # Scale the input data using the loaded StandardScaler
        input_scaled = scaler.transform(input_encoded)

        # Make predictions using the selected model
        selected_model = request.form.get('model_name').lower()  # Get selected model name
        model = None  # Initialize the model variable

        if selected_model == 'xgboost regressor':
            dinput = xgb.DMatrix(input_scaled)
            model = xgb_model
            prediction = model.predict(dinput)
        elif selected_model == 'support vector machine (svm)':
            model = svm_model
            prediction = model.predict(input_scaled)
        elif selected_model == 'ridge regression':
            model = ridge_model
            prediction = model.predict(input_scaled)
        elif selected_model == 'random forest regressor':
            model = rf_model
            prediction = model.predict(input_scaled)
        elif selected_model == 'mlp regressor':
            model = mlp_model
            prediction = model.predict(input_scaled)
        elif selected_model == 'lightgbm regressor':
            model = lgbm_model
            prediction = model.predict(input_scaled)
        elif selected_model == 'lasso regression':
            model = lasso_model
            prediction = model.predict(input_scaled)
        elif selected_model == 'k-nearest neighbors (knn)':
            model = knn_model
            prediction = model.predict(input_scaled)
        elif selected_model == 'catboost regressor':
            model = catboost_model
            prediction = model.predict(input_scaled)

        if model is None:
            return render_template('index.html', feature_names=feature_names, error_message="Invalid model selected")

        # Display the prediction on a separate page
        return render_template('result.html', prediction=prediction[0])
    
    except Exception as e:
        return render_template('index.html', feature_names=feature_names, error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
