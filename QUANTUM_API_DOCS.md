# Quantum Meta Fraud Detection API Documentation

## Overview

The Quantum Meta Fraud Detection API is an advanced fraud detection system that combines quantum computing, classical machine learning, and AI-powered logical analysis to provide comprehensive fraud scoring for financial transactions.

## Features

- **Quantum + Classical Fusion**: Combines quantum SVM, XGBoost, and rule-based models
- **AI-Powered Analysis**: Integrates Gemini AI for logical fraud pattern detection
- **Real-time Processing**: Fast API responses with sub-second analysis
- **Batch Processing**: Analyze multiple transactions simultaneously
- **Comprehensive Scoring**: Detailed fraud scores with confidence metrics
- **Risk Assessment**: Automated risk level classification and action recommendations

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements_quantum_api.txt

# Install additional quantum dependencies (optional)
pip install pennylane
```

### 2. Start the API Server

```bash
# Method 1: Direct start
python quantum_fraud_api.py

# Method 2: Using startup script
python start_quantum_api.py

# Method 3: Using uvicorn directly
uvicorn quantum_fraud_api:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the API

- **API Base URL**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## API Endpoints

### Health & Status

#### `GET /health`
Check API health status

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00",
  "version": "1.0.0"
}
```

#### `GET /model/stats`
Get model statistics and configuration

**Response:**
```json
{
  "model_version": "quantum-meta-v1.0",
  "meta_weights": {
    "quantum_weight": 0.25,
    "classical_weight": 0.25,
    "neuro_fusion_weight": 0.30,
    "gemini_logical_weight": 0.20
  },
  "risk_thresholds": {
    "low": 25,
    "medium": 50,
    "high": 75,
    "critical": 90
  },
  "neuro_qkad_loaded": true,
  "gemini_available": true,
  "components": [
    "Quantum SVM (4-qubit)",
    "Classical XGBoost",
    "Rule-based Logic",
    "Gemini AI Analysis",
    "Meta-fusion Layer"
  ]
}
```

### Transaction Analysis

#### `POST /analyze`
Comprehensive fraud analysis for a single transaction

**Request Body:**
```json
{
  "amount": 85000,
  "hour_of_day": 2,
  "is_weekend": 1,
  "day_of_week": "Saturday",
  "sender_age_group": "18-25",
  "receiver_age_group": "46-55",
  "sender_state": "Delhi",
  "sender_bank": "HDFC",
  "receiver_bank": "Unknown Bank",
  "merchant_category": "Entertainment",
  "device_type": "Android",
  "transaction_type": "P2P",
  "network_type": "WiFi",
  "transaction_status": "SUCCESS"
}
```

**Response:**
```json
{
  "fraud_score": 67.5,
  "risk_level": "MEDIUM",
  "recommended_action": "ADDITIONAL VERIFICATION - Medium risk detected",
  "individual_scores": {
    "quantum_score": 25.0,
    "classical_score": 30.0,
    "neuro_qkad_fusion_score": 43.3,
    "gemini_logical_score": 75.0
  },
  "confidence_score": 61.1,
  "model_agreement": 80.5,
  "uncertainty_measure": 38.9,
  "primary_risk_factors": [
    "Late night transaction",
    "High amount transaction",
    "Cross-bank transaction"
  ],
  "fraud_type_detected": "Rule-based analysis",
  "analysis_timestamp": "2024-01-15T02:30:45.123456",
  "model_version": "quantum-meta-v1.0",
  "processing_time_ms": 245.67
}
```

#### `POST /score`
Quick fraud score (simplified response)

**Request Body:** Same as `/analyze`

**Response:**
```json
{
  "fraud_score": 67.5,
  "risk_level": "MEDIUM",
  "timestamp": "2024-01-15T02:30:45.123456"
}
```

#### `POST /analyze/batch`
Batch analysis for multiple transactions

**Request Body:**
```json
{
  "transactions": [
    {
      "amount": 5000,
      "hour_of_day": 10,
      "is_weekend": 0,
      "day_of_week": "Monday"
    },
    {
      "amount": 75000,
      "hour_of_day": 2,
      "is_weekend": 1,
      "day_of_week": "Sunday"
    }
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "fraud_score": 25.3,
      "risk_level": "LOW",
      // ... full analysis for each transaction
    }
  ],
  "summary": {
    "total_transactions": 2,
    "successful_analyses": 2,
    "failed_analyses": 0,
    "average_fraud_score": 46.15,
    "high_risk_count": 1,
    "processing_time_ms": 456.78,
    "timestamp": "2024-01-15T02:30:45.123456"
  }
}
```

### Utility Endpoints

#### `GET /example`
Get example transaction data for testing

#### `GET /`
API information and available endpoints

## Request Parameters

### Required Fields
- **amount** (float): Transaction amount (≥ 0)
- **hour_of_day** (int): Hour of day (0-23)

### Optional Fields (with defaults)
- **is_weekend** (int): Weekend flag (0 or 1) - Default: 0
- **day_of_week** (string): Day name - Default: "Monday"
- **sender_age_group** (string): Age group - Default: "26-35"
- **receiver_age_group** (string): Age group - Default: "26-35"
- **sender_state** (string): State name - Default: "Delhi"
- **sender_bank** (string): Bank name - Default: "SBI"
- **receiver_bank** (string): Bank name - Default: "SBI"
- **merchant_category** (string): Category - Default: "Other"
- **device_type** (string): Device type - Default: "Android"
- **transaction_type** (string): Transaction type - Default: "P2P"
- **network_type** (string): Network type - Default: "4G"
- **transaction_status** (string): Status - Default: "SUCCESS"

### Valid Values

**Age Groups**: `18-25`, `26-35`, `36-45`, `46-55`, `56+`

**Device Types**: `Android`, `iOS`, `Web`, `Other`

**Transaction Types**: `P2P`, `P2M`, `M2P`, `Other`

**Network Types**: `4G`, `5G`, `WiFi`, `3G`, `Other`

**Days**: `Monday`, `Tuesday`, `Wednesday`, `Thursday`, `Friday`, `Saturday`, `Sunday`

## Response Format

### Fraud Score
- **Range**: 0-100
- **0-25**: Minimal risk
- **25-50**: Low risk
- **50-75**: Medium risk
- **75-90**: High risk
- **90-100**: Critical risk

### Risk Levels
- **MINIMAL**: Very low fraud probability
- **LOW**: Low fraud probability
- **MEDIUM**: Medium fraud probability
- **HIGH**: High fraud probability
- **CRITICAL**: Very high fraud probability

### Recommended Actions
- **APPROVE**: Minimal fraud risk
- **MONITOR TRANSACTION**: Low risk, continue monitoring
- **ADDITIONAL VERIFICATION**: Medium risk detected
- **MANUAL REVIEW REQUIRED**: High fraud probability
- **BLOCK TRANSACTION**: Critical fraud risk detected

## Usage Examples

### Python Example

```python
import requests

# Single transaction analysis
transaction = {
    "amount": 50000,
    "hour_of_day": 14,
    "is_weekend": 0,
    "day_of_week": "Tuesday",
    "sender_age_group": "26-35",
    "receiver_age_group": "26-35"
}

response = requests.post("http://localhost:8000/analyze", json=transaction)
result = response.json()

print(f"Fraud Score: {result['fraud_score']}%")
print(f"Risk Level: {result['risk_level']}")
print(f"Action: {result['recommended_action']}")
```

### cURL Example

```bash
# Quick score
curl -X POST "http://localhost:8000/score" \
     -H "Content-Type: application/json" \
     -d '{
       "amount": 25000,
       "hour_of_day": 10,
       "is_weekend": 0,
       "day_of_week": "Monday"
     }'
```

### JavaScript Example

```javascript
const transaction = {
  amount: 75000,
  hour_of_day: 22,
  is_weekend: 1,
  day_of_week: "Saturday",
  sender_age_group: "18-25",
  receiver_age_group: "56+"
};

fetch('http://localhost:8000/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(transaction)
})
.then(response => response.json())
.then(data => {
  console.log('Fraud Score:', data.fraud_score);
  console.log('Risk Level:', data.risk_level);
});
```

## Performance

- **Single Transaction**: ~200-500ms
- **Batch Processing**: ~50-100ms per transaction
- **Concurrent Requests**: Supports multiple simultaneous requests
- **Throughput**: 100+ requests per second (depending on hardware)

## Error Handling

### HTTP Status Codes
- **200**: Success
- **400**: Bad Request (invalid input)
- **422**: Validation Error
- **500**: Internal Server Error
- **503**: Service Unavailable (model not loaded)

### Error Response Format
```json
{
  "detail": "Error description",
  "error_type": "ValidationError",
  "timestamp": "2024-01-15T02:30:45.123456"
}
```

## Testing

### Run Test Suite
```bash
# Start the API server first
python quantum_fraud_api.py

# In another terminal, run tests
python test_quantum_api.py
```

### Manual Testing
1. Visit `http://localhost:8000/docs` for interactive API documentation
2. Use the example endpoint to get sample data
3. Test with different transaction scenarios

## Deployment

### Production Deployment
```bash
# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn quantum_fraud_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_quantum_api.txt .
RUN pip install -r requirements_quantum_api.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "quantum_fraud_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Security Considerations

1. **API Keys**: Store Gemini API keys securely (environment variables)
2. **Rate Limiting**: Implement rate limiting for production use
3. **Authentication**: Add authentication for production deployment
4. **Input Validation**: All inputs are validated using Pydantic models
5. **CORS**: Configure CORS settings for production

## Monitoring & Logging

- All requests are logged with timestamps
- Processing times are tracked
- Model performance metrics are available
- Health check endpoint for monitoring

## Support

For issues or questions:
1. Check the interactive documentation at `/docs`
2. Review the test examples in `test_quantum_api.py`
3. Check logs for detailed error information

## Version History

- **v1.0.0**: Initial release with quantum meta model integration
- Features: Single/batch analysis, comprehensive scoring, AI integration