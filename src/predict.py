import joblib
import numpy as np
import os

# Get correct path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

lr = joblib.load(os.path.join(BASE_DIR, "models", "linear.pkl"))
rf = joblib.load(os.path.join(BASE_DIR, "models", "rf.pkl"))

def predict_price(model_name, features):
    features = np.array(features).reshape(1, -1)

    if model_name == "Linear Regression":
        return lr.predict(features)[0]
    else:
        return rf.predict(features)[0]