# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title='Stock Price Prediction',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ===========================
# CUSTOM STYLING
# ===========================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ===========================
# HEADER
# ===========================
st.markdown('<h1 class="main-header">📈 Stock Price Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown('#### Powered by Linear Regression & Real-Time Market Data')
st.divider()

# ===========================
# SIDEBAR CONFIGURATION
# ===========================
with st.sidebar:
    st.header('⚙️ Configuration')
    st.markdown('---')
    
    # Stock Selection
    st.subheader('📊 Stock Selection')
    ticker = st.text_input(
        'Enter Stock Ticker',
        value='AAPL',
        help='Examples: AAPL, TSLA, GOOGL, MSFT',
        placeholder='e.g., AAPL'
    ).upper()
    
    st.markdown('---')
    
    # Date Range
    st.subheader('📅 Date Range')
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            'Start Date',
            value=pd.to_datetime('2020-01-01'),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            'End Date',
            value=pd.to_datetime('2023-12-31'),
            max_value=datetime.now()
        )
    
    st.markdown('---')
    
    # Model Options
    st.subheader('🤖 Model Options')
    test_size = st.slider(
        'Test Set Size (%)',
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        help='Percentage of data used for testing'
    )
    
    retrain = st.checkbox(
        'Retrain Model',
        value=True,
        help='Train a new model with the fetched data'
    )
    
    st.markdown('---')
    run_button = st.button('🚀 Run Prediction', type='primary', use_container_width=True)
    
    st.markdown('---')
    st.caption('💡 **Tip**: Use at least 2 years of data for better predictions')

# ===========================
# HELPER FUNCTIONS
# ===========================
@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker, start, end):
    """Fetch stock data from Yahoo Finance"""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return None, "No data available for this ticker/date range"
        return df, None
    except Exception as e:
        return None, str(e)

def prepare_features(df):
    """Prepare feature matrix and target vector"""
    X = df[['Open', 'High', 'Low', 'Volume']].copy()
    y = df['Close'].copy()
    return X, y

def train_model(X_train, y_train):
    """Train linear regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def calculate_metrics(y_true, y_pred):
    """Calculate model performance metrics"""
    return {
        'R² Score': r2_score(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }

def create_prediction_plot(y_test, y_pred, ticker):
    """Create interactive prediction visualization"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Actual vs Predicted Close Price', 'Prediction Error'),
        vertical_spacing=0.12,
        row_heights=[0.7, 0.3]
    )
    
    # Main prediction plot
    fig.add_trace(
        go.Scatter(
            x=y_test.index,
            y=y_test,
            mode='lines',
            name='Actual',
            line=dict(color='#2E86AB', width=2),
            hovertemplate='<b>Date</b>: %{x}<br><b>Actual</b>: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=y_test.index,
            y=y_pred,
            mode='lines',
            name='Predicted',
            line=dict(color='#A23B72', width=2, dash='dash'),
            hovertemplate='<b>Date</b>: %{x}<br><b>Predicted</b>: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Error plot
    errors = y_test.values - y_pred
    fig.add_trace(
        go.Scatter(
            x=y_test.index,
            y=errors,
            mode='lines',
            name='Error',
            line=dict(color='#F18F01', width=1.5),
            fill='tozeroy',
            hovertemplate='<b>Date</b>: %{x}<br><b>Error</b>: $%{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Error ($)", row=2, col=1)
    
    fig.update_layout(
        title=dict(
            text=f'<b>{ticker} Stock Price Prediction Analysis</b>',
            x=0.5,
            xanchor='center'
        ),
        height=700,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    return fig

# ===========================
# MAIN APPLICATION LOGIC
# ===========================
if run_button or 'data_loaded' not in st.session_state:
    # Validate inputs
    if not ticker:
        st.error('⚠️ Please enter a valid stock ticker')
        st.stop()
    
    if start_date >= end_date:
        st.error('⚠️ Start date must be before end date')
        st.stop()
    
    # Fetch data
    with st.spinner(f'🔄 Fetching data for **{ticker}**...'):
        df, error = fetch_stock_data(ticker, start_date, end_date)
    
    if error:
        st.error(f'❌ Failed to fetch data: {error}')
        st.stop()
    
    if df is None or df.empty:
        st.warning('⚠️ No data returned. Try another ticker or date range.')
        st.stop()
    
    # Store in session state
    st.session_state.data_loaded = True
    st.session_state.df = df
    st.session_state.ticker = ticker
    
    st.success(f'✅ Successfully loaded {len(df)} days of data for **{ticker}**')

# Continue only if data is loaded
if 'data_loaded' in st.session_state and st.session_state.data_loaded:
    df = st.session_state.df
    ticker = st.session_state.ticker
    
    # ===========================
    # DATA OVERVIEW
    # ===========================
    st.subheader('📊 Data Overview')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Total Records', len(df))
    with col2:
        st.metric('Date Range', f"{len(df)} days")
    with col3:
        latest_close = float(df['Close'].iloc[-1])
        st.metric("Latest Close", f"${latest_close:.2f}")

    with col4:
    # Safely calculate total return
    try:
        if df.empty or df['Close'].isna().all():
            change = 0.0
        else:
            first_close = df['Close'].iloc[0] if not pd.isna(df['Close'].iloc[0]) else 0
            last_close = df['Close'].iloc[-1] if not pd.isna(df['Close'].iloc[-1]) else 0
            if first_close == 0:
                change = 0.0
            else:
                change = ((last_close - first_close) / first_close) * 100
    except Exception:
        change = 0.0

    st.metric('Total Return', f"{change:.2f}%", delta=f"{change:.2f}%")

    
    with st.expander('📋 View Raw Data (Last 10 Rows)', expanded=False):
        st.dataframe(df.tail(10), use_container_width=True)
    
    st.divider()
    
    # ===========================
    # MODEL TRAINING & PREDICTION
    # ===========================
    st.subheader('🤖 Model Training & Predictions')
    
    with st.spinner('Training model...'):
        # Prepare data
        X, y = prepare_features(df)
        
        # Split data
        test_ratio = test_size / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, shuffle=False
        )
        
        # Train model
        if retrain:
            model = train_model(X_train, y_train)
        else:
            st.info('ℹ️ Using existing model (if available)')
            model = LinearRegression().fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
    
    # ===========================
    # METRICS DISPLAY
    # ===========================
    st.markdown('### 📈 Model Performance Metrics')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        r2_color = 'normal' if metrics['R² Score'] > 0.7 else 'inverse'
        st.metric('R² Score', f"{metrics['R² Score']:.4f}", 
                 help='Coefficient of determination (closer to 1 is better)')
    
    with col2:
        st.metric('RMSE', f"${metrics['RMSE']:.2f}",
                 help='Root Mean Squared Error (lower is better)')
    
    with col3:
        st.metric('MAE', f"${metrics['MAE']:.2f}",
                 help='Mean Absolute Error (lower is better)')
    
    with col4:
        accuracy = (1 - metrics['MAE'] / y_test.mean()) * 100
        st.metric('Accuracy', f"{accuracy:.2f}%",
                 help='Relative accuracy based on MAE')
    
    # Performance interpretation
    if metrics['R² Score'] > 0.8:
        st.success('🎉 Excellent model performance!')
    elif metrics['R² Score'] > 0.6:
        st.info('✅ Good model performance')
    else:
        st.warning('⚠️ Model performance could be improved. Consider using more data or features.')
    
    st.divider()
    
    # ===========================
    # VISUALIZATION
    # ===========================
    st.subheader('📉 Prediction Visualization')
    fig = create_prediction_plot(y_test, y_pred, ticker)
    st.plotly_chart(fig, use_container_width=True)
    
    # ===========================
    # FEATURE IMPORTANCE
    # ===========================
    st.subheader('🔍 Feature Importance')
    feature_importance = pd.DataFrame({
        'Feature': ['Open', 'High', 'Low', 'Volume'],
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    fig_importance = go.Figure(go.Bar(
        x=feature_importance['Coefficient'],
        y=feature_importance['Feature'],
        orientation='h',
        marker=dict(color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    ))
    
    fig_importance.update_layout(
        title='Feature Coefficients in Linear Regression Model',
        xaxis_title='Coefficient Value',
        yaxis_title='Feature',
        height=300,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)
    
    st.divider()
    
    # ===========================
    # DOWNLOAD PREDICTIONS
    # ===========================
    st.subheader('💾 Export Predictions')
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'Date': y_test.index,
        'Actual_Price': y_test.values,
        'Predicted_Price': y_pred,
        'Error': y_test.values - y_pred,
        'Percent_Error': ((y_test.values - y_pred) / y_test.values) * 100
    })
    
    csv = predictions_df.to_csv(index=False)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.download_button(
            label='📥 Download Predictions (CSV)',
            data=csv,
            file_name=f'{ticker}_predictions_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
            use_container_width=True
        )
    with col2:
        st.metric('Records', len(predictions_df))
    
    with st.expander('👀 Preview Predictions', expanded=False):
        st.dataframe(predictions_df.head(10), use_container_width=True)

else:
    st.info('👈 Configure your settings in the sidebar and click **Run Prediction** to begin')
    
    # Show example
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
st.caption('⚠️ **Disclaimer**: This tool is for educational purposes only. Not financial advice. Past performance does not guarantee future results.')
st.caption('📊 Data source: Yahoo Finance | Model: Scikit-learn Linear Regression')


