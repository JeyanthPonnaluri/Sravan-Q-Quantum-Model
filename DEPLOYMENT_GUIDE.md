# 🚀 Deployment Guide - Neuro-QKAD

## 📋 Table of Contents
1. [Development Deployment](#development-deployment)
2. [Production Deployment](#production-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Monitoring & Logging](#monitoring--logging)
6. [Security Considerations](#security-considerations)
7. [Performance Optimization](#performance-optimization)
8. [Backup & Recovery](#backup--recovery)

---

## 🛠️ Development Deployment

### Local Development Setup
```bash
# Clone repository
git clone <repository-url>
cd neuro-qkad

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements_simple.txt

# Train models
python enhanced_fraud_detector.py

# Start development server
python enhanced_api.py
```

### Development Configuration
```python
# enhanced_api.py - Development settings
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="127.0.0.1",  # Local only
        port=8001,
        reload=True,       # Auto-reload on changes
        log_level="debug"  # Verbose logging
    )
```

---

## 🏭 Production Deployment

### Production Server Setup

#### 1. System Requirements
- **OS**: Ubuntu 20.04+ / CentOS 8+ / Windows Server 2019+
- **Python**: 3.8+
- **RAM**: 8GB+ (for quantum simulations)
- **CPU**: 4+ cores
- **Storage**: 10GB+ free space

#### 2. Production Installation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3.9 python3.9-venv python3.9-dev -y

# Create application user
sudo useradd -m -s /bin/bash fraudapp
sudo su - fraudapp

# Setup application
git clone <repository-url> /home/fraudapp/neuro-qkad
cd /home/fraudapp/neuro-qkad

python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements_simple.txt

# Train models
python enhanced_fraud_detector.py
```

#### 3. Production Configuration
```python
# production_config.py
import os

class ProductionConfig:
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", 8001))
    WORKERS = int(os.getenv("WORKERS", 4))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
    MODELS_PATH = os.getenv("MODELS_PATH", "enhanced_models/")
    MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", 1024))
```

#### 4. Systemd Service
```ini
# /etc/systemd/system/fraud-detection.service
[Unit]
Description=Fraud Detection API
After=network.target

[Service]
Type=exec
User=fraudapp
Group=fraudapp
WorkingDirectory=/home/fraudapp/neuro-qkad
Environment=PATH=/home/fraudapp/neuro-qkad/venv/bin
ExecStart=/home/fraudapp/neuro-qkad/venv/bin/python enhanced_api.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable fraud-detection
sudo systemctl start fraud-detection
sudo systemctl status fraud-detection
```

#### 5. Nginx Reverse Proxy
```nginx
# /etc/nginx/sites-available/fraud-detection
server {
    listen 80;
    server_name your-domain.com;
    
    client_max_body_size 1M;
    
    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Static files (if any)
    location /static/ {
        alias /home/fraudapp/neuro-qkad/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/fraud-detection /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_simple.txt .
RUN pip install --no-cache-dir -r requirements_simple.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 fraudapp && chown -R fraudapp:fraudapp /app
USER fraudapp

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Start application
CMD ["python", "enhanced_api.py"]
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  fraud-detection:
    build: .
    ports:
      - "8001:8001"
    environment:
      - LOG_LEVEL=info
      - WORKERS=4
    volumes:
      - ./enhanced_models:/app/enhanced_models:ro
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - fraud-detection
    restart: unless-stopped

volumes:
  models:
  logs:
```

### Build and Deploy
```bash
# Build image
docker build -t fraud-detection:latest .

# Run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs fraud-detection
```

---

## ☁️ Cloud Deployment

### AWS Deployment

#### 1. EC2 Instance
```bash
# Launch EC2 instance (t3.large recommended)
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type t3.large \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxxx \
    --subnet-id subnet-xxxxxxxxx \
    --user-data file://user-data.sh
```

#### 2. Application Load Balancer
```bash
# Create ALB
aws elbv2 create-load-balancer \
    --name fraud-detection-alb \
    --subnets subnet-xxxxxxxx subnet-yyyyyyyy \
    --security-groups sg-xxxxxxxxx

# Create target group
aws elbv2 create-target-group \
    --name fraud-detection-targets \
    --protocol HTTP \
    --port 8001 \
    --vpc-id vpc-xxxxxxxxx \
    --health-check-path /health
```

#### 3. Auto Scaling Group
```bash
# Create launch template
aws ec2 create-launch-template \
    --launch-template-name fraud-detection-template \
    --launch-template-data file://launch-template.json

# Create auto scaling group
aws autoscaling create-auto-scaling-group \
    --auto-scaling-group-name fraud-detection-asg \
    --launch-template LaunchTemplateName=fraud-detection-template \
    --min-size 2 \
    --max-size 10 \
    --desired-capacity 3 \
    --target-group-arns arn:aws:elasticloadbalancing:...
```

### Google Cloud Platform

#### 1. Cloud Run Deployment
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/fraud-detection

# Deploy to Cloud Run
gcloud run deploy fraud-detection \
    --image gcr.io/PROJECT-ID/fraud-detection \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 10
```

#### 2. Kubernetes Engine
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detection
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-detection
  template:
    metadata:
      labels:
        app: fraud-detection
    spec:
      containers:
      - name: fraud-detection
        image: gcr.io/PROJECT-ID/fraud-detection:latest
        ports:
        - containerPort: 8001
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: fraud-detection-service
spec:
  selector:
    app: fraud-detection
  ports:
  - port: 80
    targetPort: 8001
  type: LoadBalancer
```

---

## 📊 Monitoring & Logging

### Application Monitoring

#### 1. Prometheus Metrics
```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Metrics
REQUEST_COUNT = Counter('fraud_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('fraud_request_duration_seconds', 'Request duration')
PREDICTION_SCORES = Histogram('fraud_prediction_scores', 'Prediction score distribution')
MODEL_LOAD_STATUS = Gauge('fraud_model_loaded', 'Model load status')

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_DURATION.observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

#### 2. Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Fraud Detection Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(fraud_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, fraud_request_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Prediction Score Distribution",
        "type": "heatmap",
        "targets": [
          {
            "expr": "fraud_prediction_scores_bucket",
            "legendFormat": "Score: {{le}}"
          }
        ]
      }
    ]
  }
}
```

### Centralized Logging

#### 1. ELK Stack Configuration
```yaml
# docker-compose-elk.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:7.15.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5044:5044"
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:7.15.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

#### 2. Structured Logging
```python
# logging_config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'prediction_data'):
            log_entry['prediction'] = record.prediction_data
            
        return json.dumps(log_entry)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fraud_detection.log')
    ]
)

for handler in logging.root.handlers:
    handler.setFormatter(JSONFormatter())
```

---

## 🔒 Security Considerations

### 1. API Security
```python
# security.py
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

@app.post("/predict")
async def predict(transaction: TransactionFull, user=Depends(verify_token)):
    # Protected endpoint
    return predict_fraud_enhanced(transaction)
```

### 2. Rate Limiting
```python
# rate_limiting.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("100/minute")
async def predict(request: Request, transaction: TransactionFull):
    return predict_fraud_enhanced(transaction)
```

### 3. Input Validation
```python
# validation.py
from pydantic import BaseModel, validator, Field

class TransactionFull(BaseModel):
    amount: float = Field(..., gt=0, le=1000000, description="Transaction amount")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of day")
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        if v > 1000000:
            raise ValueError('Amount too large')
        return v
    
    @validator('sender_bank', 'receiver_bank')
    def validate_banks(cls, v):
        allowed_banks = ['SBI', 'HDFC', 'ICICI', 'Axis', 'PNB', 'Kotak']
        if v not in allowed_banks:
            raise ValueError(f'Bank must be one of {allowed_banks}')
        return v
```

### 4. HTTPS Configuration
```nginx
# nginx-ssl.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/ssl/certs/your-domain.crt;
    ssl_certificate_key /etc/ssl/private/your-domain.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    
    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

---

## ⚡ Performance Optimization

### 1. Model Caching
```python
# model_cache.py
from functools import lru_cache
import hashlib
import json

@lru_cache(maxsize=1000)
def cached_prediction(transaction_hash: str):
    # Cache predictions for identical transactions
    return predict_fraud_enhanced(transaction)

def get_transaction_hash(transaction: dict) -> str:
    # Create hash of transaction for caching
    transaction_str = json.dumps(transaction, sort_keys=True)
    return hashlib.md5(transaction_str.encode()).hexdigest()
```

### 2. Async Processing
```python
# async_processing.py
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.post("/predict")
async def predict_async(transaction: TransactionFull):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor, 
        predict_fraud_enhanced, 
        transaction
    )
    return result
```

### 3. Database Connection Pooling
```python
# db_pool.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

---

## 💾 Backup & Recovery

### 1. Model Backup Strategy
```bash
#!/bin/bash
# backup_models.sh

BACKUP_DIR="/backup/fraud-detection"
MODEL_DIR="/app/enhanced_models"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR/$DATE"

# Backup models
cp -r "$MODEL_DIR" "$BACKUP_DIR/$DATE/"

# Compress backup
tar -czf "$BACKUP_DIR/models_$DATE.tar.gz" -C "$BACKUP_DIR" "$DATE"

# Remove uncompressed backup
rm -rf "$BACKUP_DIR/$DATE"

# Keep only last 30 backups
find "$BACKUP_DIR" -name "models_*.tar.gz" -mtime +30 -delete

echo "Backup completed: models_$DATE.tar.gz"
```

### 2. Automated Backup with Cron
```bash
# Add to crontab
0 2 * * * /home/fraudapp/backup_models.sh >> /var/log/model_backup.log 2>&1
```

### 3. Recovery Procedure
```bash
#!/bin/bash
# restore_models.sh

BACKUP_FILE="$1"
MODEL_DIR="/app/enhanced_models"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop service
sudo systemctl stop fraud-detection

# Backup current models
mv "$MODEL_DIR" "${MODEL_DIR}.old"

# Extract backup
tar -xzf "$BACKUP_FILE" -C /tmp/
mv /tmp/enhanced_models "$MODEL_DIR"

# Start service
sudo systemctl start fraud-detection

echo "Models restored from $BACKUP_FILE"
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy Fraud Detection

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements_simple.txt
        pip install pytest
    - name: Run tests
      run: pytest tests/ -v

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to production
      run: |
        # Deploy script here
        ./deploy.sh
```

---

## 📋 Deployment Checklist

### Pre-deployment
- [ ] Models trained and validated
- [ ] All tests passing
- [ ] Security configurations reviewed
- [ ] SSL certificates installed
- [ ] Monitoring setup configured
- [ ] Backup strategy implemented

### Deployment
- [ ] Application deployed
- [ ] Health checks passing
- [ ] Load balancer configured
- [ ] DNS records updated
- [ ] SSL/TLS working
- [ ] Monitoring active

### Post-deployment
- [ ] Smoke tests completed
- [ ] Performance metrics normal
- [ ] Logs flowing correctly
- [ ] Alerts configured
- [ ] Documentation updated
- [ ] Team notified

---

**Deployment Guide Version**: 1.0  
**Last Updated**: January 2025  
**Next Review**: March 2025