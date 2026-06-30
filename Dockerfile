FROM python:3.11-slim

# Set up a non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install
COPY --chown=user:user requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY --chown=user:user . /app/

# Create necessary directories so the app can write to them
RUN mkdir -p /app/datasets /app/models /app/logs /app/reports

# Expose port 7860 which is required by Hugging Face Spaces
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
