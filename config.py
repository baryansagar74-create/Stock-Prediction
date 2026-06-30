import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    # App Config
    APP_NAME: str = "Stock Market Price Prediction ANN"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # MLflow Config
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "stock_prediction_ann"
    
    # Base Directories
    BASE_DIR: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = BASE_DIR / "datasets"
    MODEL_DIR: Path = BASE_DIR / "models"
    LOG_DIR: Path = BASE_DIR / "logs"
    REPORT_DIR: Path = BASE_DIR / "reports"
    DB_PATH: Path = BASE_DIR / "predictions.db"
    
    # Training Config
    TRAIN_EPOCHS: int = 100
    TRAIN_BATCH_SIZE: int = 32
    TRAIN_VALIDATION_SPLIT: float = 0.2
    TRAIN_LOOK_BACK: int = 60
    TRAIN_LEARNING_RATE: float = 0.001
    
    # Feature Config
    FEATURES: List[str] = [
        'Close', 'Open', 'High', 'Low', 'Volume',
        'Returns', 'SMA_20', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal',
        'Bollinger_Upper', 'Bollinger_Lower', 'Volatility', 'Volume_Change'
    ]
    TARGET_COL: str = 'Close'

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra='ignore')

# Initialize settings
settings = Settings()

# Setup directories
for d in [settings.DATA_DIR, settings.MODEL_DIR, settings.LOG_DIR, settings.REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
