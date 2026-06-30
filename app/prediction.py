import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from app.preprocessing import download_data, preprocess_data
from app.database import save_prediction
from config import settings
from app.utils import logger

def predict_future_price(ticker: str) -> dict:
    """Predicts the next future price based on recent live data."""
    try:
        model_path = settings.MODEL_DIR / f"ann_model_{ticker}.keras"
        scaler_path = settings.MODEL_DIR / f"scaler_{ticker}.pkl"
        
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"Model or scaler for {ticker} not found. Please train the model first.")
            
        # Load Model & Scaler
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        # We need recent data (e.g., 6 months is enough for a 60-day look_back + indicator windows)
        df = download_data(ticker, period="6mo")
        scaled_data, target_idx, df_processed = preprocess_data(df, ticker, fit_scaler=False)
        
        look_back = settings.TRAIN_LOOK_BACK
        if len(scaled_data) < look_back:
            raise ValueError(f"Not enough data to create sequence. Need {look_back}, got {len(scaled_data)}.")
            
        # Get the latest sequence
        recent_sequence = scaled_data[-look_back:]
        X_input = np.expand_dims(recent_sequence, axis=0) # Shape: (1, look_back, features)
        
        # Predict
        y_pred_scaled = model.predict(X_input)
        
        # Unscale
        dummy_pred = np.zeros((1, scaled_data.shape[1]))
        dummy_pred[0, target_idx] = y_pred_scaled[0, 0]
        predicted_price = scaler.inverse_transform(dummy_pred)[0, target_idx]
        
        # Calculate % change
        last_actual_price = df_processed.iloc[-1]['Close']
        change_pct = ((predicted_price - last_actual_price) / last_actual_price) * 100
        
        # Simple Confidence Interval Estimation (e.g., +/- 1.5% based on typical MSE, this is just an estimate)
        # To be statistically rigorous, we should compute std of residuals, but here we approximate based on volatility.
        # Volatility is usually computed in preprocessing
        last_volatility = df_processed.iloc[-1]['Volatility']
        ci_lower = float(predicted_price - last_volatility)
        ci_upper = float(predicted_price + last_volatility)
        
        # Save to DB
        save_prediction(
            ticker=ticker,
            prediction=float(predicted_price),
            change_pct=float(change_pct),
            ci=[ci_lower, ci_upper],
            actual_price=float(last_actual_price)
        )
        
        logger.info(f"Prediction for {ticker} completed: ${predicted_price:.2f}")
        
        return {
            "predicted_price": float(predicted_price),
            "current_price": float(last_actual_price),
            "change_pct": float(change_pct),
            "confidence_interval": [ci_lower, ci_upper]
        }
        
    except Exception as e:
        logger.error(f"Error predicting future price: {e}")
        raise
