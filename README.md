# Super Market Sales Prediction

> A machine learning project for predicting super market sales using various regression models.

---

## Table of Contents

- [Description](#description)
- [How to Use](#how-to-use)
- [Models and Evaluation](#models-and-evaluation)
- [Histogram](#histogram)
- [Residual Plots](#residual-plots)
- [Scatter Plots](#scatter-plots)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

---

## Description

The "Super Market Sales Prediction" project is a machine learning-based application that predicts super market sales based on various features such as branch, city, customer type, gender, product line, unit price, quantity, tax, cost of goods sold, gross income, and customer rating. This project demonstrates the use of multiple regression algorithms to make sales predictions.

Key features of the project include:
- Preprocessing and encoding of categorical features.
- Training and evaluation of different regression models.
- Visualization of model performance.

---

## How to Use

To use this project, follow these steps:

1. Clone the repository to your local machine:

   ```shell
   git clone https://github.com/pypi-ahmad/Super-Market-Sales-Prediction.git
   ```

2. Install the required dependencies:

   ```shell
   pip install -r requirements.txt
   ```

3. Run the Flask application:

   ```shell
   python app.py
   ```

4. Open a web browser and go to `http://localhost:5000` to access the prediction form.

5. Select a regression model, enter the required input values, and click "Predict."

6. View the prediction results on the next page.

---

## Models and Evaluation

The project utilizes various regression models, including:
- XGBoost Regressor
- Support Vector Machine (SVM)
- Ridge Regression
- Random Forest Regressor
- MLP Regressor
- LightGBM Regressor
- Lasso Regression
- K-Nearest Neighbors (KNN)
- CatBoost Regressor

Model performance is evaluated using metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

![Comparison of MAE Values for Different Models](Comparison%20of%20MAE%20Values%20for%20Different%20Models.png)

![Comparison of RMSE Values for Different Models](Comparison%20of%20RMSE%20Values%20for%20Different%20Models.png)

---

## Histogram

Histograms of residuals for different models provide insights into the distribution of prediction errors.

1. CatBoost Residual Histogram
   ![CatBoost Residual Histogram](Histogram/catboost_residuals_histogram.png)

2. KNN Residual Histogram
   ![KNN Residual Histogram](Histogram/knn_residuals_histogram.png)

3. Lasso Residual Histogram
   ![Lasso Residual Histogram](Histogram/lasso_residuals_histogram.png)

4. LightGBM Residual Histogram
   ![LightGBM Residual Histogram](Histogram/lgbm_residuals_histogram.png)

5. MLP Residual Histogram
   ![MLP Residual Histogram](Histogram/mlp_residuals_histogram.png)

6. Random Forest Residual Histogram
   ![Random Forest Residual Histogram](Histogram/rf_residuals_histogram.png)

7. Ridge Residual Histogram
   ![Ridge Residual Histogram](Histogram/ridge_residuals_histogram.png)

8. SVM Residual Histogram
   ![SVM Residual Histogram](Histogram/svm_residuals_histogram.png)

9. XGBoost Residual Histogram
   ![XGBoost Residual Histogram](Histogram/xgb_residuals_histogram.png)

---

## Residual Plots

Residual plots for different models help visualize the relationship between predicted and actual values.

1. CatBoost Residual Plot
   ![CatBoost Residual Plot](Residual%20Plots/catboost_residual_plot.png)

2. KNN Residual Plot
   ![KNN Residual Plot](Residual%20Plots/knn_residual_plot.png)

3. Lasso Residual Plot
   ![Lasso Residual Plot](Residual%20Plots/lasso_residual_plot.png)

4. LightGBM Residual Plot
   ![LightGBM Residual Plot](Residual%20Plots/lgbm_residual_plot.png)

5. MLP Residual Plot
   ![MLP Residual Plot](Residual%20Plots/mlp_residual_plot.png)

6. Random Forest Residual Plot
   ![Random Forest Residual Plot](Residual%20Plots/rf_residual_plot.png)

7. Ridge Residual Plot
   ![Ridge Residual Plot](Residual%20Plots/ridge_residual_plot.png)

8. SVM Residual Plot
   ![SVM Residual Plot](Residual%20Plots/svm_residual_plot.png)

9. XGBoost Residual Plot
   ![XGBoost Residual Plot](Residual%20Plots/xgb_residual_plot.png)

---

## Scatter Plots

Scatter plots for different models show the distribution of predicted values against actual values.

1. CatBoost Scatter Plot
   ![CatBoost Scatter Plot](Scatter%20Plots/catboost_scatter.png)

2. KNN Scatter Plot
   ![KNN Scatter Plot](Scatter%20Plots/knn_scatter.png)

3. Lasso Scatter Plot
   ![Lasso

 Scatter Plot](Scatter%20Plots/lasso_scatter.png)

4. LightGBM Scatter Plot
   ![LightGBM Scatter Plot](Scatter%20Plots/lightgbm_scatter.png)

5. MLP Scatter Plot
   ![MLP Scatter Plot](Scatter%20Plots/mlp_scatter.png)

6. Random Forest Scatter Plot
   ![Random Forest Scatter Plot](Scatter%20Plots/rf_scatter.png)

7. Ridge Scatter Plot
   ![Ridge Scatter Plot](Scatter%20Plots/ridge_scatter.png)

8. SVM Scatter Plot
   ![SVM Scatter Plot](Scatter%20Plots/svm_scatter.png)

9. XGBoost Scatter Plot
   ![XGBoost Scatter Plot](Scatter%20Plots/xgb_scatter.png)

---

## Installation

To run this project locally, ensure you have Python and Git installed. Follow the steps in the "How to Use" section above.

---

## Contributing

Contributions are welcome! If you would like to contribute to this project, please fork the repository and create a pull request. Feel free to open issues for bug reports or feature requests.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
