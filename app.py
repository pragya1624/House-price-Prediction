import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.predict import predict_price
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# Title
# -------------------------------
st.title("🏠 House Price Prediction")
st.write("Compare Linear Regression and Random Forest")

# -------------------------------
# User Input
# -------------------------------
area = st.number_input("Enter Area")
bedrooms = st.number_input("Enter Bedrooms")

# Model selection
model = st.selectbox(
    "Select Model",
    ["Linear Regression", "Random Forest"]
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):
    features = [area, bedrooms]
    result = predict_price(model, features)

    st.success(f"💰 Predicted Price: {result}")

# -------------------------------
# Data Visualization (Area vs Price)
# -------------------------------
st.subheader("📊 House Price vs Area")

df = pd.read_csv("data/data.csv")

fig1, ax1 = plt.subplots()
ax1.scatter(df['area'], df['price'])

ax1.set_xlabel("Area")
ax1.set_ylabel("Price")
ax1.set_title("House Price vs Area")

st.pyplot(fig1)

# -------------------------------
# Model Comparison
# -------------------------------
if st.button("Compare Models"):

    X = df[['area', 'bedrooms']]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load models
    lr = joblib.load("models/linear.pkl")
    rf = joblib.load("models/rf.pkl")

    def evaluate(model):
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        r2 = r2_score(y_test, pred)
        return mae, rmse, r2

    # Evaluate both models
    lr_mae, lr_rmse, lr_r2 = evaluate(lr)
    rf_mae, rf_rmse, rf_r2 = evaluate(rf)

    # Create dataframe
    data = {
        "Model": ["Linear Regression", "Random Forest"],
        "RMSE": [lr_rmse, rf_rmse],
        "R2 Score": [lr_r2, rf_r2]
    }

    df_results = pd.DataFrame(data)

    # Show table
    st.subheader("📊 Model Comparison")
    st.dataframe(df_results)

    # Best model
    best_model = df_results.loc[df_results["RMSE"].idxmin()]["Model"]
    st.success(f"🏆 Best Model: {best_model}")

    # -------------------------------
    # Graph (RMSE Comparison)
    # -------------------------------
    st.subheader("📈 Model Performance (RMSE Comparison)")

    fig2, ax2 = plt.subplots()
    ax2.bar(df_results["Model"], df_results["RMSE"])

    ax2.set_ylabel("RMSE")
    ax2.set_title("Model Comparison")

    st.pyplot(fig2)