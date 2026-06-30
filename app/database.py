import sqlite3
import json
from datetime import datetime
from config import settings
from app.utils import logger

def init_db():
    """Initializes the SQLite database with required tables."""
    try:
        conn = sqlite3.connect(settings.DB_PATH)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction REAL NOT NULL,
            prediction_change_pct REAL,
            confidence_interval TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            actual_price REAL,
            model_version TEXT
        )
        ''')
        
        # Metrics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            rmse REAL,
            mae REAL,
            mape REAL,
            r2_score REAL,
            directional_accuracy REAL,
            hit_ratio REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_version TEXT
        )
        ''')
        
        # Training Logs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            epochs INTEGER,
            batch_size INTEGER,
            final_loss REAL,
            final_val_loss REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
    finally:
        if conn:
            conn.close()

def save_prediction(ticker: str, prediction: float, change_pct: float, ci: list, actual_price: float = None, version: str = "v1.0"):
    """Saves a prediction record to the database."""
    try:
        conn = sqlite3.connect(settings.DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO predictions (ticker, prediction, prediction_change_pct, confidence_interval, actual_price, model_version, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (ticker, prediction, change_pct, json.dumps(ci), actual_price, version, datetime.now()))
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")
    finally:
        if conn:
            conn.close()

def get_prediction_history(ticker: str, limit: int = 50):
    """Retrieves prediction history for a ticker."""
    try:
        conn = sqlite3.connect(settings.DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
        SELECT * FROM predictions 
        WHERE ticker = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
        ''', (ticker, limit))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return []
    finally:
        if conn:
            conn.close()

def save_metrics(ticker: str, metrics: dict, version: str = "v1.0"):
    """Saves evaluation metrics."""
    try:
        conn = sqlite3.connect(settings.DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO metrics (ticker, rmse, mae, mape, r2_score, directional_accuracy, hit_ratio, timestamp, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticker, 
            metrics.get("RMSE"), 
            metrics.get("MAE"), 
            metrics.get("MAPE"), 
            metrics.get("R2_Score"), 
            metrics.get("Directional_Accuracy"), 
            metrics.get("Hit_Ratio"), 
            datetime.now(),
            version
        ))
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
    finally:
        if conn:
            conn.close()

def get_latest_metrics(ticker: str):
    """Gets the most recent metrics for a ticker."""
    try:
        conn = sqlite3.connect(settings.DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
        SELECT * FROM metrics 
        WHERE ticker = ? 
        ORDER BY timestamp DESC 
        LIMIT 1
        ''', (ticker,))
        row = cursor.fetchone()
        return dict(row) if row else None
    except Exception as e:
        logger.error(f"Error fetching latest metrics: {e}")
        return None
    finally:
        if conn:
            conn.close()

def save_training_log(ticker: str, epochs: int, batch_size: int, final_loss: float, final_val_loss: float):
    """Saves a training log."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO training_logs (ticker, epochs, batch_size, final_loss, final_val_loss, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (ticker, epochs, batch_size, final_loss, final_val_loss, datetime.now()))
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving training log: {e}")
    finally:
        if conn:
            conn.close()
            
def get_training_logs(ticker: str, limit: int = 10):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
        SELECT * FROM training_logs 
        WHERE ticker = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
        ''', (ticker, limit))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching training logs: {e}")
        return []
    finally:
        if conn:
            conn.close()
