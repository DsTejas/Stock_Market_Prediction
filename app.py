# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title='Stock Price Prediction',
    page_icon='📈',
    layout='wide'
)

# Title
st.title('📈 Stock Price Prediction Dashboard')
st.markdown('#### Powered by Linear Regression & Real-Time Market Data')
st.divider()

# Sidebar
with st.sidebar:
    st.header('⚙️ Configuration')
    st.markdown('---')
    
    st.subheader('📊 Stock Selection')
    ticker = st.text_input('Enter Stock Ticker', value='AAPL', help='Examples: AAPL, TSLA, GOOGL, MSFT').upper()
    
    st.markdown('---')
    st.subheader('📅 Date Range')
    start_date = st.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
    end_date = st.date_input('End Date', value=pd.to_datetime('2023-12-31'))
    
    st.markdown('---')
    st.subheader('🤖 Model Options')
    test_size = st.slider('Test Set Size (%)', min_value=10, max_value=40, value=20, step=5)
    retrain = st.checkbox('Retrain Model', value=True)
    
    st.markdown('---')
    run_button = st.button('🚀 Run Prediction', type='primary', use_container_width=True)

# Functions
@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return None, "No data available"
        return df, None
    except Exception as e:
        return None, str(e)

# Main logic
if run_button:
    if not ticker:
        st.error('⚠️ Please enter a valid stock ticker')
        st.stop()
    
    if start_date >= end_date:
        st.error('⚠️ Start date must be before end date')
        st.stop()
    
    with st.spinner(f'🔄 Fetching data for **{ticker}**...'):
        df, error = fetch_stock_data(ticker, start_date, end_date)
    
    if error or df is None:
        st.error(f'❌ Failed to fetch data: {error}')
        st.stop()
    
    st.success(f'✅ Successfully loaded {len(df)} days of data for **{ticker}**')
    
    # Data Overview
    st.subheader('📊 Data Overview')
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric('Total Records', len(df))
    with col2:
        st.metric('Date Range', f"{len(df)} days")
    with col3:
        st.metric('Latest Close', f"${df['Close'].iloc[-1]:.2f}")
    with col4:
        change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
        st.metric('Total Return', f"{change:.2f}%")
    
    with st.expander('📋 View Raw Data (Last 10 Rows)'):
        st.dataframe(df.tail(10), use_container_width=True)
    
    st.divider()
    
    # Prepare features
    X = df[['Open', 'High', 'Low', 'Volume']].copy()
    y = df['Close'].copy()
    
    # Train/test split
    test_ratio = test_size / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    
    # Train model
    st.subheader('🤖 Model Training & Predictions')
    with st.spinner('Training model...'):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Display metrics
    st.markdown('### 📈 Model Performance Metrics')
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric('R² Score', f"{r2:.4f}")
    with col2:
        st.metric('RMSE', f"${rmse:.2f}")
    with col3:
        st.metric('MAE', f"${mae:.2f}")
    with col4:
        accuracy = (1 - mae / y_test.mean()) * 100
        st.metric('Accuracy', f"{accuracy:.2f}%")
    
    if r2 > 0.8:
        st.success('🎉 Excellent model performance!')
    elif r2 > 0.6:
        st.info('✅ Good model performance')
    else:
        st.warning('⚠️ Model performance could be improved')
    
    st.divider()
    
    # Visualization
    st.subheader('📉 Prediction Visualization')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=y_test.values,
        mode='lines',
        name='Actual',
        line=dict(color='#2E86AB', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=y_pred,
        mode='lines',
        name='Predicted',
        line=dict(color='#A23B72', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'{ticker} Stock Price: Actual vs Predicted',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader('🔍 Feature Importance')
    feature_df = pd.DataFrame({
        'Feature': ['Open', 'High', 'Low', 'Volume'],
        'Coefficient': model.coef_
    })
    
    fig2 = go.Figure(go.Bar(
        x=feature_df['Coefficient'],
        y=feature_df['Feature'],
        orientation='h',
        marker=dict(color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    ))
    
    fig2.update_layout(
        title='Feature Coefficients',
        xaxis_title='Coefficient Value',
        yaxis_title='Feature',
        height=300,
        template='plotly_white'
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    st.divider()
    
    # Download predictions
    st.subheader('💾 Export Predictions')
    
    predictions_df = pd.DataFrame({
        'Date': y_test.index.strftime('%Y-%m-%d'),
        'Actual_Price': y_test.values,
        'Predicted_Price': y_pred,
        'Error': y_test.values - y_pred,
        'Percent_Error': ((y_test.values - y_pred) / y_test.values) * 100
    })
    
    csv = predictions_df.to_csv(index=False)
    
    st.download_button(
        label='📥 Download Predictions (CSV)',
        data=csv,
        file_name=f'{ticker}_predictions_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv',
        use_container_width=True
    )
    
    with st.expander('👀 Preview Predictions'):
        st.dataframe(predictions_df.head(10), use_container_width=True)

else:
    st.info('👈 Configure your settings in the sidebar and click **Run Prediction** to begin')
    
    st.markdown('### 🎯 What This App Does')
    st.markdown("""
    This application uses **Linear Regression** to predict stock prices based on:
    - Opening price
    - Daily high
    - Daily low
    - Trading volume
    
    **How to use:**
    1. Enter a stock ticker (e.g., AAPL, TSLA, GOOGL)
    2. Select your date range
    3. Configure model options
    4. Click **Run Prediction**
    """)

# Footer
st.divider()
st.caption('⚠️ **Disclaimer**: This tool is for educational purposes only. Not financial advice.')
st.caption('📊 Data source: Yahoo Finance | Model: Scikit-learn Linear Regression')
