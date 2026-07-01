FROM python:3.12-slim

WORKDIR /app

# Install build-essential for any compilation requirements
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies in correct order
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port (7860 is default for Hugging Face Spaces, or use PORT environment variable)
EXPOSE 7860

# Start command
CMD ["sh", "-c", "uvicorn government_fraud_system:app --host 0.0.0.0 --port ${PORT:-7860}"]
