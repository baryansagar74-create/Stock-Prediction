import logging
from logging.handlers import RotatingFileHandler
import os
import csv
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fpdf import FPDF
from config import settings

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Configures and returns a logger instance with rotating file handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO if not settings.DEBUG else logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Rotating File handler (10 MB per file, max 5 files)
        log_file = settings.LOG_DIR / "app.log"
        fh = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Stream handler
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        
    return logger

logger = get_logger("utils")

# ---------------------------------------------------------
# Metrics Calculation
# ---------------------------------------------------------
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculates evaluation metrics."""
    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        epsilon = 1e-10
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        
        r2 = r2_score(y_true, y_pred)
        
        if len(y_true) > 1:
            actual_diff = np.diff(y_true.flatten())
            pred_diff = np.diff(y_pred.flatten())
            dir_acc = np.mean((actual_diff * pred_diff) > 0) * 100
        else:
            dir_acc = 0.0
            
        hit_ratio = dir_acc
        
        metrics = {
            "RMSE": float(rmse),
            "MAE": float(mae),
            "MAPE": float(mape),
            "R2_Score": float(r2),
            "Directional_Accuracy": float(dir_acc),
            "Hit_Ratio": float(hit_ratio)
        }
        return metrics
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {}

# ---------------------------------------------------------
# Report Generation
# ---------------------------------------------------------
def generate_csv_report(ticker: str, data: list) -> str:
    """Generates a CSV report for predictions."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ticker}_prediction_report_{timestamp}.csv"
        filepath = settings.REPORT_DIR / filename
        
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Date", "Actual Price", "Predicted Price", "Timestamp"])
            for row in data:
                writer.writerow(row)
                
        logger.info(f"CSV report generated: {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Error generating CSV: {e}")
        return ""

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(80)
        self.cell(30, 10, 'Stock Prediction Evaluation Report', 0, 0, 'C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(ticker: str, metrics: dict, prediction: float) -> str:
    """Generates a PDF evaluation report."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ticker}_evaluation_report_{timestamp}.pdf"
        filepath = settings.REPORT_DIR / filename
        
        pdf = PDFReport()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        pdf.cell(200, 10, txt=f"Ticker: {ticker}", ln=1, align='L')
        pdf.cell(200, 10, txt=f"Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align='L')
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Evaluation Metrics", ln=1, align='L')
        pdf.set_font("Arial", size=12)
        
        for k, v in metrics.items():
            pdf.cell(200, 10, txt=f"{k}: {v:.4f}", ln=1, align='L')
            
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Latest Prediction", ln=1, align='L')
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Predicted Next Price: ${prediction:.2f}", ln=1, align='L')
        
        pdf.output(str(filepath))
        logger.info(f"PDF report generated: {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        return ""
