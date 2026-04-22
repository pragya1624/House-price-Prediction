from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset
data = fetch_california_housing()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target
df["price"] = data.target

# Save to CSV
df.to_csv("../data/data.csv", index=False)

print("✅ Big dataset created successfully!")