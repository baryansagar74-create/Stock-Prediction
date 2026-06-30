import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import hashlib
import json
from config import settings
from app.utils import logger
import warnings

warnings.filterwarnings('ignore')

def download_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Downloads historical data from yfinance."""
    try:
        logger.info(f"Downloading data for {ticker} (Period: {period}, Interval: {interval})")
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
            
        df.reset_index(inplace=True)
        # Handle different timezone formats
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        elif 'Datetime' in df.columns:
            df['Date'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)
            df.drop('Datetime', axis=1, inplace=True)
            
        return df
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Computes technical indicators for feature engineering."""
    try:
        # Returns
        df['Returns'] = df['Close'].pct_change()
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['Bollinger_Mid'] = df['Close'].rolling(window=20).mean()
        df['Bollinger_Std'] = df['Close'].rolling(window=20).std()
        df['Bollinger_Upper'] = df['Bollinger_Mid'] + (df['Bollinger_Std'] * 2)
        df['Bollinger_Lower'] = df['Bollinger_Mid'] - (df['Bollinger_Std'] * 2)
        
        # Volatility & Volume Change
        df['Volatility'] = df['Close'].rolling(window=20).std()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        return df
    except Exception as e:
        logger.error(f"Error computing indicators: {e}")
        raise

def preprocess_data(df: pd.DataFrame, ticker: str, fit_scaler: bool = True) -> tuple:
    """Preprocesses the DataFrame, removing NaNs and scaling features."""
    try:
        df = compute_technical_indicators(df)
        
        # Drop rows with NaN values created by indicators
        df.dropna(inplace=True)
        
        # Remove duplicates
        df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
        
        # Outlier Detection/Capping (Simple IQR method)
        for col in settings.FEATURES:
            if col in df.columns:
                Q1 = df[col].quantile(0.05)
                Q3 = df[col].quantile(0.95)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        
        # Save preprocessed data with versioning
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        data_filename = f"{ticker}_processed_{timestamp}.csv"
        data_path = settings.DATA_DIR / data_filename
        df.to_csv(data_path, index=False)
        
        # Calculate Checksum
        with open(data_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()
            
        logger.info(f"Dataset saved: {data_filename} (SHA256: {checksum})")
        
        # Scaling
        feature_data = df[settings.FEATURES].values
        
        scaler_path = settings.MODEL_DIR / f"scaler_{ticker}.pkl"
        if fit_scaler:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(feature_data)
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved for {ticker}")
        else:
            scaler = joblib.load(scaler_path)
            scaled_data = scaler.transform(feature_data)
            
        # Target column index
        target_idx = settings.FEATURES.index(settings.TARGET_COL)
            
        return scaled_data, target_idx, df, data_filename, checksum
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

def create_sequences(data: np.ndarray, target_idx: int, look_back: int = 60) -> tuple:
    """Creates sliding window sequences for ANN input."""
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)
