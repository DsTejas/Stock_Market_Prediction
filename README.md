# Stock Price Prediction

This project predicts stock closing prices using **Linear Regression** and visualizes the results with Python libraries.  
Data is collected from Yahoo Finance, and the model is trained to forecast future prices.

## Features
- Fetches stock data automatically using `yfinance`
- Trains a Linear Regression model
- Visualizes actual vs predicted prices
- Shows regression line on historical data
- Predicts the next day’s stock price

## Tech Stack
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- yFinance  

## Installation
```bash
git clone <your-repo-link>
cd stock-price-prediction
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
