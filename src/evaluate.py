import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("../data/data.csv")

X = df[['area', 'bedrooms']]
y = df['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load models
lr = joblib.load("../models/linear.pkl")
rf = joblib.load("../models/rf.pkl")

def evaluate(model):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, pred)

    return mae, rmse, r2

# Evaluate both
lr_mae, lr_rmse, lr_r2 = evaluate(lr)
rf_mae, rf_rmse, rf_r2 = evaluate(rf)

print("\n📊 MODEL COMPARISON:\n")

print("Linear Regression")
print("MAE:", lr_mae)
print("RMSE:", lr_rmse)
print("R2:", lr_r2)
print("-" * 30)

print("Random Forest")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2:", rf_r2)