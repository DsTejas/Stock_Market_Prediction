# notebook.py

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Fetch stock data
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
data.reset_index(inplace=True)

# Feature selection
data['Day'] = np.arange(len(data))
X = data[['Day']]
y = data['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Visualization 1 - Actual vs Predicted
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Closing Price")
plt.show()

# Visualization 2 - Regression line on all data
plt.figure(figsize=(10,5))
sns.scatterplot(x=X['Day'], y=y, label="Data")
sns.lineplot(x=X['Day'], y=model.predict(X), color="red", label="Regression Line")
plt.title("Regression Line on Stock Data")
plt.show()

# Predict next day
next_day = np.array([[len(data)]])
predicted_price = model.predict(next_day)[0]
print("Predicted next day price:", predicted_price)
