import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def train_models():
    # Get the repo root (two levels up from src/)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load dataset
    data_path = os.path.join(BASE_DIR, "data", "data.csv")
    df = pd.read_csv(data_path)

    # Features and target
    X = df.drop("price", axis=1)
    y = df["price"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize models
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train models
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Save models to repo root /models/
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(lr, os.path.join(models_dir, "linear.pkl"))
    joblib.dump(rf, os.path.join(models_dir, "rf.pkl"))

    print("✅ Models trained and saved successfully!")