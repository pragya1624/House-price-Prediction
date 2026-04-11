import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("data.csv")

# Features (input) and target (output)
X = data[['area', 'bedrooms']]
y = data['price']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
import pandas as pd

new_data = pd.DataFrame([[2000, 3]], columns=['area', 'bedrooms'])
prediction = model.predict(new_data)

print("Predicted House Price:", prediction[0])
import matplotlib.pyplot as plt

plt.scatter(data['area'], data['price'])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Price vs Area")
plt.savefig("graph.png")
plt.show()

import pandas as pd

area = float(input("Enter area: "))
bedrooms = int(input("Enter bedrooms: "))

new_data = pd.DataFrame([[area, bedrooms]], columns=['area', 'bedrooms'])

pred = model.predict(new_data)

print("Estimated Price:", pred[0])