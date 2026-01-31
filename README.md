# Universal Forecasting Lab 🔮

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FLAML](https://img.shields.io/badge/FLAML-Microsoft-blue?style=for-the-badge)](https://github.com/microsoft/FLAML)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

A comprehensive, interactive research lab for time-series forecasting. This tool democratizes AI by allowing anyone to upload sales data and automatically train state-of-the-art machine learning models (XGBoost, LightGBM, Random Forest) using Microsoft's FLAML.

## 📋 Executive Summary
The **Universal Forecasting Lab** transforms raw CSV data into actionable insights. It handles the complexity of data science—preprocessing, feature engineering, model selection, and hyperparameter tuning—automatically. 

Whether you are predicting supermarket sales, stock prices, or website traffic, this lab provides a "Deep Data Scan" to understand your data and a "Model Arena" to find the most accurate algorithm for your specific problem.

## ✨ Key Features

### 🚀 AutoML Engine
Powered by **Microsoft FLAML**, the engine automatically:
- Engineers features (Lags, Rolling Means, Date Parts).
- Tests multiple algorithms (LGBM, XGBoost, ExtraTrees).
- Optimizes hyperparameters for the best RMSE.

### 📊 Deep EDA (Exploratory Data Analysis)
Understand your data before you model it:
- **Decomposition**: Automatically separates Trend, Seasonality, and Noise.
- **Autocorrelation**: Visualizes lag dependencies (ACF/PACF).
- **Correlation Heatmaps**: Identifies key drivers of your target variable.

### ⚔️ Model Battle
See exactly what happened behind the scenes:
- **Leaderboard**: A transparent ranking of all models tried.
- **Feature Importance**: Discover which variables matter most (e.g., "Is 'Hour' more important than 'Month'?").
- **Training History**: Visualize the loss minimization curve.

### 🔮 Future Forecast
- **Interactive Scenario Planning**: Use a slider to forecast 1-30 days ahead.
- **Recursive Forecasting**: Automatically generates future features to project trajectories.

## 🛠️ Quick Start

### 1. Installation
Clone the repository and install the dependencies.
```bash
pip install -r requirements.txt
```

### 2. Train Demo Models
Run the training script to verify the installation and generate the initial model bundle using the included `supermarket_sales.csv`.
```bash
python train_forecast.py
```

### 3. Launch the Lab
Start the interactive dashboard.
```bash
streamlit run app.py
```

## 📂 Project Structure
- `app.py`: The Streamlit dashboard application.
- `train_forecast.py`: The core `ForecastingEngine` class and training logic.
- `supermarket_sales.csv`: Demo dataset.
- `requirements.txt`: Project dependencies.

---
*Built with ❤️ for Data Science enthusiasts.*
