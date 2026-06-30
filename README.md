---
title: Stock Prediction
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# Stock Market Price Prediction Using ANN on Live Data

[![Live Demo on Hugging Face Spaces](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Spaces-blue?logo=huggingface)](https://huggingface.co/spaces/HYPERx54/Stock-Prediction)

This project is a complete, production-quality web application designed for predicting stock market prices using an Artificial Neural Network (ANN) trained on live data fetched from yfinance. It is built as an industrial-grade system suitable for academic research and portfolio showcase.

## Architecture

1. **Frontend**: HTML5, CSS3 (Bootstrap 5, Glassmorphism design), Vanilla JS, Chart.js for interactive visualizations.
2. **Backend**: FastAPI for robust, high-performance REST APIs.
3. **Database**: SQLite for storing historical predictions, metrics, and training logs.
4. **Machine Learning Pipeline**:
   - **Data Source**: Live data fetched from Yahoo Finance (`yfinance`).
   - **Preprocessing**: Handles missing values, performs sliding window sequence generation, computes 15+ technical indicators (RSI, MACD, Bollinger Bands, EMA), and applies MinMaxScaler.
   - **Model**: Multi-layer Feedforward Neural Network using `TensorFlow`/`Keras` optimized with Adam and dropout layers to prevent overfitting.
   - **Evaluation**: Computes RMSE, MAE, MAPE, R² Score, and Directional Accuracy.
5. **Reporting**: Automated PDF and CSV report generation using `fpdf2` and Python `csv`.

## Requirements

Ensure you have Python 3.12+ installed. Install the requirements using:

```bash
pip install -r requirements.txt
```

## Running the Application

To run the application locally:

```bash
python run.py
```

This script will automatically start the Uvicorn server and open the web dashboard in your default browser at `http://127.0.0.1:8000`.

## API Endpoints

- `POST /api/train`: Triggers asynchronous model training for a given ticker.
- `POST /api/predict`: Generates the next price prediction using the trained model.
- `GET /api/history/{ticker}`: Retrieves the historical predictions stored in the database.
- `GET /api/live/{ticker}`: Fetches the current real-time market data.
- `GET /api/metrics/{ticker}`: Returns the evaluation metrics for the latest trained model.
- `GET /api/training-logs/{ticker}`: Returns epoch-by-epoch training logs.
- `POST /api/download-report`: Generates and downloads a comprehensive PDF evaluation report.
- `POST /api/download-csv`: Generates and downloads the prediction history as a CSV file.

## Workflow

1. Enter any valid Yahoo Finance ticker (e.g., `AAPL`, `NVDA`, `RELIANCE.NS`) in the web interface.
2. Click **Train Model**. This will fetch up to 5 years of historical data, compute indicators, split chronologically, and train the ANN.
3. Click **Predict** to forecast the next closing price. The dashboard will automatically update with charts comparing historical actual prices with the predicted trajectory.
4. Export PDF or CSV reports directly from the sidebar.
5. Toggle "Auto Refresh" on the dashboard to automatically poll yfinance and generate new predictions every minute.

## Future Improvements

- Integration with Celery/Redis for distributed training queues.
- Support for LSTM/GRU advanced architectures alongside the Feedforward ANN.
- Real-time WebSocket streaming for millisecond tick data.
