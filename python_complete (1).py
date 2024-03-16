import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
img = Image.open('Capture.PNG')
st.image(img,width=250)

# Load the dataset (assuming you have your data in a CSV file)
data = pd.read_csv("Advertising.csv")

X = data[['TV', 'Newspaper', 'Radio']]
y = data['Sales']


st.subheader("Dataset (Sample 5 rows)")
st.write(data.head(5))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(),
    "Ridge Regression": Ridge(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor()
}

trained_models = {}
model_rmse = {}
model_adj_r2 = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    y_pred = model.predict(X_test)
    model_rmse[name] = np.sqrt(mean_squared_error(y_test, y_pred))
    n = len(X_test)
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2_score(y_test, y_pred)) * ((n - 1) / (n - p - 1))
    model_adj_r2[name] = adj_r2

# Streamlit UI
st.title("Sales Prediction App")
st.write("Predict sales based on advertising spending")

TV = st.text_input("Enter TV budget and press Enter:")
Newspaper = st.text_input("Enter Newspaper budget and press Enter:")
Radio = st.text_input("Enter Radio budget and press Enter:")

if TV and Newspaper and Radio:
    st.subheader("Model Predictions:")
    for model_name, model in trained_models.items():
        prediction = model.predict([[float(TV), float(Newspaper), float(Radio)]])
        st.sidebar.subheader(model_name)
        st.sidebar.write(f"Predicted Sales: {prediction[0]}")
        st.sidebar.write(f"RMSE: {model_rmse[model_name]}")
        st.sidebar.write(f"Adjusted R2: {model_adj_r2[model_name]}")

    # Plot R2 scores for models
    st.subheader("Adjusted R2 Score Comparison")
    fig_adj_r2 = plt.figure(figsize=(10, 6))
    plt.bar(model_adj_r2.keys(), model_adj_r2.values(), color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Adjusted R2 Score')
    plt.title('Adjusted R2 Score for Different Models')
    plt.xticks(rotation=45)
    st.pyplot(fig_adj_r2)

    # Plot correlation between features and target
    st.subheader("Correlation between Features and Sales")
    fig_correlation, axes_correlation = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(X.columns):
        sns.scatterplot(x=X[col], y=y, ax=axes_correlation[i])
        axes_correlation[i].set_xlabel(col)
        axes_correlation[i].set_ylabel('Sales')
    st.pyplot(fig_correlation)