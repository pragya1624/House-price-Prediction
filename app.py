import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("data.csv")

# Prepare data
X = data[['area', 'bedrooms']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# UI
st.title("🏠 House Price Prediction App")

st.write("Enter details to predict house price")

# User input
area = st.number_input("Enter Area (sq ft)", value=1000)
bedrooms = st.number_input("Enter Number of Bedrooms", value=2)

# Prediction button
if st.button("Predict Price"):
    new_data = pd.DataFrame([[area, bedrooms]], columns=['area', 'bedrooms'])
    prediction = model.predict(new_data)

    st.success(f"Estimated Price: ₹{prediction[0]:,.2f}")