import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction App")

# Load dataset
df = pd.read_csv("data/data.csv")

# Split features and target
X = df.drop("price", axis=1)
y = df["price"]

# Load models
linear_model = joblib.load("models/linear.pkl")
rf_model = joblib.load("models/rf.pkl")

# -------------------------
# MODEL SELECTION
# -------------------------
model_choice = st.selectbox("Select Model", ["Linear Regression", "Random Forest"])

# -------------------------
# PREDICTION (using sample input)
# -------------------------
st.subheader("🔮 Predict House Price")

if st.button("Predict Price"):
    # Use first row as sample input (temporary solution)
    features = X.iloc[0].values.reshape(1, -1)

    if model_choice == "Linear Regression":
        prediction = linear_model.predict(features)
    else:
        prediction = rf_model.predict(features)

    st.success(f"💰 Predicted Price: {prediction[0]:.2f}")

# -------------------------
# MODEL COMPARISON
# -------------------------
st.subheader("📊 Model Performance (RMSE Comparison)")

if st.button("Compare Models"):
    y_pred_lr = linear_model.predict(X)
    y_pred_rf = rf_model.predict(X)

    rmse_lr = np.sqrt(mean_squared_error(y, y_pred_lr))
    rmse_rf = np.sqrt(mean_squared_error(y, y_pred_rf))

    df_results = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "RMSE": [rmse_lr, rmse_rf]
    })

    # Show best model
    best_model = df_results.loc[df_results["RMSE"].idxmin()]

    st.success(f"🏆 Best Model: {best_model['Model']}")

    # Show table
    st.write(df_results)

    # Plot comparison
    fig, ax = plt.subplots()
    ax.bar(df_results["Model"], df_results["RMSE"])
    ax.set_ylabel("RMSE")
    ax.set_title("Model Comparison")
    st.pyplot(fig)

# -------------------------
# DATA VISUALIZATION
# -------------------------
st.subheader("📈 Data Visualization")

feature_for_graph = st.selectbox(
    "Select Feature for Graph",
    X.columns
)

fig2, ax2 = plt.subplots()
ax2.scatter(df[feature_for_graph], df["price"])
ax2.set_xlabel(feature_for_graph)
ax2.set_ylabel("Price")
ax2.set_title(f"{feature_for_graph} vs Price")

st.pyplot(fig2)