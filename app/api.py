from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os
from fastapi.responses import FileResponse
from app.training import train_model
from app.prediction import predict_future_price
from app.database import get_prediction_history, get_latest_metrics, get_training_logs
from app.utils import generate_csv_report, generate_pdf_report, logger
import yfinance as yf

router = APIRouter()

class TickerRequest(BaseModel):
    ticker: str

@router.post("/train")
async def api_train_model(request: TickerRequest, background_tasks: BackgroundTasks):
    """Initiates model training."""
    try:
        # In a real production system, this should be an async task queue (e.g., Celery)
        # For this project, we'll run it in FastAPI's background tasks
        background_tasks.add_task(train_model, request.ticker.upper())
        return {"status": "success", "message": f"Training started for {request.ticker.upper()}"}
    except Exception as e:
        logger.error(f"Error in /train API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict")
async def api_predict(request: TickerRequest):
    """Predicts future price for a trained model."""
    try:
        result = predict_future_price(request.ticker.upper())
        return {"status": "success", "data": result}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in /predict API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{ticker}")
async def api_history(ticker: str, limit: int = 50):
    """Retrieves prediction history."""
    try:
        history = get_prediction_history(ticker.upper(), limit)
        return {"status": "success", "data": history}
    except Exception as e:
        logger.error(f"Error in /history API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/live/{ticker}")
async def api_live(ticker: str):
    """Fetches current live stock information."""
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        # Sometimes 'regularMarketPrice' or 'currentPrice' is used
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        previous_close = info.get('previousClose', 0)
        
        data = {
            "current_price": current_price,
            "previous_close": previous_close,
            "day_high": info.get('dayHigh', 0),
            "day_low": info.get('dayLow', 0),
            "open": info.get('regularMarketOpen', 0),
            "volume": info.get('volume', 0),
            "company_name": info.get('longName', ticker.upper())
        }
        return {"status": "success", "data": data}
    except Exception as e:
        logger.error(f"Error in /live API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/{ticker}")
async def api_metrics(ticker: str):
    """Retrieves latest evaluation metrics."""
    try:
        metrics = get_latest_metrics(ticker.upper())
        if not metrics:
            return {"status": "error", "message": "No metrics found. Train the model first."}
        return {"status": "success", "data": metrics}
    except Exception as e:
        logger.error(f"Error in /metrics API: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
@router.get("/training-logs/{ticker}")
async def api_training_logs(ticker: str):
    """Retrieves training logs."""
    try:
        logs = get_training_logs(ticker.upper())
        return {"status": "success", "data": logs}
    except Exception as e:
        logger.error(f"Error in /training-logs API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/download-report")
async def api_download_report(request: TickerRequest):
    """Generates and returns a PDF report."""
    try:
        ticker = request.ticker.upper()
        metrics = get_latest_metrics(ticker)
        if not metrics:
            raise HTTPException(status_code=404, detail="No metrics found.")
            
        # Get latest prediction
        history = get_prediction_history(ticker, limit=1)
        pred_val = history[0]['prediction'] if history else 0.0
        
        filepath = generate_pdf_report(ticker, metrics, pred_val)
        if not filepath or not os.path.exists(filepath):
            raise HTTPException(status_code=500, detail="Failed to generate PDF.")
            
        return FileResponse(filepath, media_type='application/pdf', filename=os.path.basename(filepath))
    except Exception as e:
        logger.error(f"Error in /download-report API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/download-csv")
async def api_download_csv(request: TickerRequest):
    """Generates and returns a CSV history report."""
    try:
        ticker = request.ticker.upper()
        history = get_prediction_history(ticker, limit=100)
        
        data = [[row['timestamp'], row['actual_price'], row['prediction'], row['timestamp']] for row in history]
        
        filepath = generate_csv_report(ticker, data)
        if not filepath or not os.path.exists(filepath):
            raise HTTPException(status_code=500, detail="Failed to generate CSV.")
            
        return FileResponse(filepath, media_type='text/csv', filename=os.path.basename(filepath))
    except Exception as e:
        logger.error(f"Error in /download-csv API: {e}")
        raise HTTPException(status_code=500, detail=str(e))
