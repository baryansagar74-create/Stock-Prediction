import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import mlflow.sklearn
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from config import settings
from app.model import build_ann_model
from app.preprocessing import download_data, preprocess_data, create_sequences
from app.database import save_training_log, save_metrics
from app.utils import logger, calculate_metrics
import datetime

def train_model(ticker: str) -> dict:
    """Trains the ANN model for the given ticker."""
    try:
        logger.info(f"Starting training pipeline for {ticker}")
        
        # 1. Data Collection & Preprocessing
        df = download_data(ticker)
        scaled_data, target_idx, df_processed, data_filename, checksum = preprocess_data(df, ticker, fit_scaler=True)
        
        # 2. Sequence Generation
        look_back = settings.TRAIN_LOOK_BACK
        X, y = create_sequences(scaled_data, target_idx, look_back)
        
        # 3. Chronological Train/Test Split (Never shuffle time-series)
        split_idx = int(len(X) * (1 - settings.TRAIN_VALIDATION_SPLIT))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # 4. Build Model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_ann_model(input_shape, learning_rate=settings.TRAIN_LEARNING_RATE)
        
        # 5. Callbacks
        model_path = settings.MODEL_DIR / f"ann_model_{ticker}.keras"
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
            TensorBoard(log_dir=settings.LOG_DIR / "tensorboard" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        ]
        
        # Initialize MLflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        
        with mlflow.start_run(run_name=f"train_{ticker}_{datetime.datetime.now().strftime('%Y%m%d')}"):
            # Log Parameters
            mlflow.log_params({
                "ticker": ticker,
                "epochs": settings.TRAIN_EPOCHS,
                "batch_size": settings.TRAIN_BATCH_SIZE,
                "look_back": settings.TRAIN_LOOK_BACK,
                "learning_rate": settings.TRAIN_LEARNING_RATE,
                "dataset_checksum": checksum,
                "dataset_filename": data_filename
            })
            
            # 6. Training
            history = model.fit(
                X_train, y_train,
                epochs=settings.TRAIN_EPOCHS,
                batch_size=settings.TRAIN_BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            shuffle=False, # Crucial for time series
            verbose=1
        )
        
            # 7. Evaluate on Test Data
            best_model_path = settings.MODEL_DIR / f"ann_model_{ticker}.keras"
            if best_model_path.exists():
                model.load_weights(best_model_path)
                
            y_pred = model.predict(X_test)
            
            # We need to unscale the predictions and targets to calculate real metrics
            import joblib
            import numpy as np
            scaler_path = settings.MODEL_DIR / f"scaler_{ticker}.pkl"
            scaler = joblib.load(scaler_path)
            
            # Create dummy arrays for inverse transform
            dummy_pred = np.zeros((len(y_pred), scaled_data.shape[1]))
            dummy_pred[:, target_idx] = y_pred.flatten()
            unscaled_pred = scaler.inverse_transform(dummy_pred)[:, target_idx]
            
            dummy_true = np.zeros((len(y_test), scaled_data.shape[1]))
            dummy_true[:, target_idx] = y_test
            unscaled_true = scaler.inverse_transform(dummy_true)[:, target_idx]
            
            # 8. Calculate Metrics
            metrics = calculate_metrics(unscaled_true, unscaled_pred)
            mlflow.log_metrics(metrics)
            
            # Log Models and Artifacts
            mlflow.keras.log_model(model, "ann_model")
            mlflow.sklearn.log_model(scaler, "scaler")
            
            # 9. Plot and Save Graphs
            plot_training_history(history, ticker)
            plot_actual_vs_predicted(unscaled_true, unscaled_pred, ticker)
            
            # Log plot artifacts
            mlflow.log_artifact(str(settings.REPORT_DIR / f"{ticker}_training_history.png"))
            mlflow.log_artifact(str(settings.REPORT_DIR / f"{ticker}_actual_vs_predicted.png"))
            
            # 10. Save Logs and Metrics to DB
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            save_training_log(ticker, len(history.history['loss']), settings.TRAIN_BATCH_SIZE, float(final_loss), float(final_val_loss))
            save_metrics(ticker, metrics)
            
            logger.info(f"Training completed and logged to MLflow for {ticker}.")
            return metrics
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

def plot_training_history(history, ticker):
    """Plots and saves training loss and MAE."""
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # MAE Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Model MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(settings.REPORT_DIR / f"{ticker}_training_history.png")
    plt.close()

def plot_actual_vs_predicted(y_true, y_pred, ticker):
    """Plots actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual Price', color='blue')
    plt.plot(y_pred, label='Predicted Price', color='red', alpha=0.7)
    plt.title(f'{ticker} - Actual vs Predicted Prices (Test Set)')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(settings.REPORT_DIR / f"{ticker}_actual_vs_predicted.png")
    plt.close()
