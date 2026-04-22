def train_models():
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("../data/data.csv")

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

# Save models
joblib.dump(lr, "../models/linear.pkl")
joblib.dump(rf, "../models/rf.pkl")

print("✅ Models trained and saved successfully!")