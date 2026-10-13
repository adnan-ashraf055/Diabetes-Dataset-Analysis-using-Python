import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'

columns = [
    'mpg',
    'cylinders',
    'displacement',
    'horsepower',
    'weight',
    'acceleration',
    'model_year',
    'origin',
    'car_name'
]

# Read dataset
df = pd.read_csv(
    url,
    sep=r'\s+',
    names=columns,
    na_values='?',
    usecols=range(8)
)

# Handle missing values
df = df.dropna()

# Display column names
print(df.columns)

# Split features and target
X = df.drop('mpg', axis=1)
y = df['mpg']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Display coefficients
print(
    f"Coefficients: "
    f"{dict(zip(X.columns, model.coef_))}"
)

print(f"Intercept: {model.intercept_:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    'r--',
    lw=2
)

plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title('Linear Regression: Actual vs Predicted MPG')

plt.show()