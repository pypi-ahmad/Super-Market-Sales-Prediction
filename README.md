# Supermarket Sales AI Command Center 🛒

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![FLAML](https://img.shields.io/badge/FLAML-AutoML-green?style=for-the-badge)](https://microsoft.github.io/FLAML/)

A full-stack AI platform designed for retail analytics, forecasting, and automated machine learning. This application empowers users to analyze massive datasets, train regression models on the fly, and simulate sales scenarios with deep explainability.

---

## 🚀 Features

### **1. 1GB Data Engine 💾**
- **Massive Uploads**: Built to handle large CSV/Excel retail datasets (up to 1GB).
- **Automated EDA**: Instantly generates distribution plots, correlation heatmaps, and categorical breakdowns.
- **Data Quality**: Automatic detection of missing values and schema inference.

### **2. AutoML Training Lab ⚙️**
- **Powered by FLAML**: State-of-the-art Automated Machine Learning from Microsoft.
- **Click-to-Train**: Select your target column, set a time budget (e.g., 60 seconds), and let the AI find the best model.
- **Performance Metrics**: Real-time tracking of R², MAE, and RMSE during training.
- **Model Persistence**: Automatically saves the best model (`automl_model.pkl`) for future use.

### **3. Sales Simulator 🔮**
- **Interactive Prediction**: Dynamic input forms generated based on your model's features.
- **Scenario Planning**: Adjust parameters like `Unit price`, `Quantity`, or `Branch` to see the impact on `Total` sales.
- **Real-Time Results**: Get instant revenue predictions.

### **4. X-Ray Vision (Explainability) 🧠**
- **SHAP Integration**: Understand *why* the model made a prediction.
- **Global Importance**: Beeswarm plots show which features drive sales the most across the dataset.
- **Local Explanation**: Waterfall plots break down how each specific input contributed to a single prediction.

---

## 🛠️ Installation & Quick Start

### Step 0: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Super-Market-Sales-Prediction.git
cd Super-Market-Sales-Prediction
```

### Option A: Docker (Recommended)
Run the application in a production-ready container.

```bash
docker-compose up --build
```
Access the app at `http://localhost:8501`.

### Option B: Local Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**
   ```bash
   streamlit run app.py
   ```

---

## 📂 Project Structure

- **`app.py`**: The main Streamlit application ("Command Center").
- **`train_automl.py`**: (Optional) Standalone script for headless model training.
- **`automl_model.pkl`**: The saved artifact containing the trained model and metadata (ignored in git).
- **`Dockerfile` / `docker-compose.yml`**: Container configuration.

---

## 📊 How to Use

1. **Select Data Source**:
   - Choose **"Train on New Data"** to upload your `supermarket_sales.csv`.
   - Or select **"Use Pre-trained Model"** if you have already trained one.

2. **Train**:
   - Go to the **AutoML Training Lab** tab.
   - Select `Total` as the target.
   - Click **🚀 Start Training**.

3. **Predict**:
   - Switch to **Sales Simulator**.
   - Tweak inputs and hit **Predict Sales**.

4. **Analyze**:
   - Check **Model Explainability** to see SHAP plots.

---

*Powered by Streamlit, FLAML, and SHAP.*

