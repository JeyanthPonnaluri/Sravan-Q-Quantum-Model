# 🔌 API Specification - Neuro-QKAD

## Base URL
```
http://127.0.0.1:8001
```

---

## 📋 Endpoints Overview

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/` | GET | Web interface | No |
| `/predict` | POST | Fraud analysis | No |
| `/health` | GET | System status | No |
| `/info` | GET | Model information | No |

---

## 🎯 POST /predict

Analyze a transaction for fraud probability using all 14 features.

### Request

**Content-Type**: `application/json`

**Body Schema**:
```json
{
  "amount": "number (required)",
  "hour_of_day": "integer (required, 0-23)",
  "is_weekend": "integer (required, 0 or 1)",
  "day_of_week": "string (required)",
  "sender_age_group": "string (required)",
  "receiver_age_group": "string (required)",
  "sender_state": "string (required)",
  "sender_bank": "string (required)",
  "receiver_bank": "string (required)",
  "merchant_category": "string (required)",
  "device_type": "string (required)",
  "transaction_type": "string (required)",
  "network_type": "string (required)",
  "transaction_status": "string (required)"
}
```

### Field Specifications

#### Numeric Fields
- **amount**: Transaction amount in INR (positive number)
- **hour_of_day**: Hour of transaction (0-23)
- **is_weekend**: Weekend flag (0=weekday, 1=weekend)

#### Categorical Fields
- **day_of_week**: `Monday`, `Tuesday`, `Wednesday`, `Thursday`, `Friday`, `Saturday`, `Sunday`
- **sender_age_group**: `18-25`, `26-35`, `36-45`, `46-55`
- **receiver_age_group**: `18-25`, `26-35`, `36-45`, `46-55`
- **sender_state**: `Delhi`, `Maharashtra`, `Karnataka`, `Uttar Pradesh`, `Tamil Nadu`, `Gujarat`, `Rajasthan`, `West Bengal`, etc.
- **sender_bank**: `SBI`, `HDFC`, `ICICI`, `Axis`, `PNB`, `Kotak`, `Yes Bank`, `IndusInd`
- **receiver_bank**: `SBI`, `HDFC`, `ICICI`, `Axis`, `PNB`, `Kotak`, `Yes Bank`, `IndusInd`
- **merchant_category**: `Grocery`, `Fuel`, `Entertainment`, `Food`, `Shopping`, `Utilities`, `Other`
- **device_type**: `Android`, `iOS`
- **transaction_type**: `P2P`, `P2M`
- **network_type**: `4G`, `5G`, `WiFi`, `3G`
- **transaction_status**: `SUCCESS`, `FAILED`

### Example Request
```bash
curl -X POST "http://127.0.0.1:8001/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "amount": 15000,
       "hour_of_day": 14,
       "is_weekend": 0,
       "day_of_week": "Friday",
       "sender_age_group": "26-35",
       "receiver_age_group": "36-45",
       "sender_state": "Delhi",
       "sender_bank": "HDFC",
       "receiver_bank": "SBI",
       "merchant_category": "Grocery",
       "device_type": "Android",
       "transaction_type": "P2M",
       "network_type": "4G",
       "transaction_status": "SUCCESS"
     }'
```

### Response

**Content-Type**: `application/json`

**Success Response (200)**:
```json
{
  "quantum_score": 12.45,
  "classical_score": 18.32,
  "logical_score": 25.00,
  "fusion_score": 16.78,
  "risk_level": "LOW",
  "confidence": "High"
}
```

**Response Schema**:
- **quantum_score**: `number` - Quantum SVM prediction (0-100%)
- **classical_score**: `number` - XGBoost prediction (0-100%)
- **logical_score**: `number` - Rule-based score (0-100%)
- **fusion_score**: `number` - Final combined score (0-100%)
- **risk_level**: `string` - Risk category (`LOW`, `LOW-MEDIUM`, `MEDIUM`, `HIGH`)
- **confidence**: `string` - Prediction confidence (`High`, `Medium`, `Low`)

### Risk Level Mapping
| Fusion Score | Risk Level | Description |
|--------------|------------|-------------|
| 0-19% | LOW | Process normally |
| 20-39% | LOW-MEDIUM | Monitor transaction |
| 40-69% | MEDIUM | Additional verification |
| 70-100% | HIGH | Immediate review required |

### Error Responses

**Validation Error (422)**:
```json
{
  "detail": [
    {
      "loc": ["body", "amount"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Server Error (500)**:
```json
{
  "detail": "Models not loaded"
}
```

---

## 🏥 GET /health

Check API and model status.

### Response

**Success Response (200)**:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "quantum_enabled": true,
  "features_count": 14
}
```

**Response Schema**:
- **status**: `string` - API status (`healthy`, `unhealthy`)
- **models_loaded**: `boolean` - Whether models are loaded
- **quantum_enabled**: `boolean` - Quantum computing availability
- **features_count**: `number` - Number of features in model

---

## ℹ️ GET /info

Get detailed model and feature information.

### Response

**Success Response (200)**:
```json
{
  "title": "Enhanced Quantum-Classical Fraud Detection",
  "description": "Uses PennyLane quantum kernel SVM + XGBoost + Rule-based logic fusion",
  "features": {
    "numeric": ["amount", "hour_of_day", "is_weekend"],
    "categorical": [
      "day_of_week", "sender_age_group", "receiver_age_group", 
      "sender_state", "sender_bank", "receiver_bank", 
      "merchant_category", "device_type", "transaction_type", 
      "network_type", "transaction_status"
    ]
  },
  "quantum_qubits": 4,
  "models": ["Quantum SVM", "XGBoost", "Logical Rules", "Fusion Model"],
  "total_features": 14
}
```

**Response Schema**:
- **title**: `string` - System title
- **description**: `string` - System description
- **features**: `object` - Feature categorization
  - **numeric**: `array` - List of numeric features
  - **categorical**: `array` - List of categorical features
- **quantum_qubits**: `number` - Number of qubits used
- **models**: `array` - List of model components
- **total_features**: `number` - Total feature count

---

## 🌐 GET /

Serves the web interface HTML page.

### Response

**Content-Type**: `text/html`

Returns a complete HTML page with:
- Interactive form for all 14 features
- Real-time fraud analysis
- Visual result display
- Responsive design

---

## 🔒 Error Handling

### HTTP Status Codes
- **200**: Success
- **422**: Validation Error (invalid input)
- **500**: Internal Server Error (model issues)

### Error Response Format
```json
{
  "detail": "Error description"
}
```

### Common Errors
1. **Missing required fields**: 422 with field details
2. **Invalid field values**: 422 with validation message
3. **Models not loaded**: 500 with "Models not loaded" message
4. **Prediction failure**: 500 with specific error details

---

## 📊 Rate Limiting

Currently no rate limiting implemented. For production:
- Recommended: 100 requests/minute per IP
- Burst: 10 requests/second
- Use Redis or similar for distributed rate limiting

---

## 🔐 Authentication

Currently no authentication required. For production:
- API Key authentication recommended
- JWT tokens for user sessions
- Role-based access control

---

## 📈 Monitoring

### Metrics to Track
- Request count by endpoint
- Response times (p50, p95, p99)
- Error rates by status code
- Prediction score distributions
- Model performance metrics

### Logging Format
```json
{
  "timestamp": "2025-01-09T14:30:00Z",
  "level": "INFO",
  "endpoint": "/predict",
  "method": "POST",
  "status_code": 200,
  "response_time_ms": 245,
  "prediction": {
    "fusion_score": 16.78,
    "risk_level": "LOW"
  }
}
```

---

## 🧪 Testing

### Unit Tests
```bash
pytest tests/test_api.py -v
```

### Integration Tests
```bash
# Test all endpoints
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/info
curl -X POST http://127.0.0.1:8001/predict -d @test_transaction.json
```

### Load Testing
```bash
# Using Apache Bench
ab -n 1000 -c 10 -T application/json -p test_transaction.json http://127.0.0.1:8001/predict
```

---

## 📚 SDK Examples

### Python SDK
```python
import requests

class FraudDetectionClient:
    def __init__(self, base_url="http://127.0.0.1:8001"):
        self.base_url = base_url
    
    def predict(self, transaction):
        response = requests.post(f"{self.base_url}/predict", json=transaction)
        return response.json()
    
    def health(self):
        response = requests.get(f"{self.base_url}/health")
        return response.json()

# Usage
client = FraudDetectionClient()
result = client.predict({
    "amount": 15000,
    "hour_of_day": 14,
    # ... other fields
})
print(f"Risk Level: {result['risk_level']}")
```

### JavaScript SDK
```javascript
class FraudDetectionAPI {
    constructor(baseUrl = 'http://127.0.0.1:8001') {
        this.baseUrl = baseUrl;
    }
    
    async predict(transaction) {
        const response = await fetch(`${this.baseUrl}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(transaction)
        });
        return response.json();
    }
    
    async health() {
        const response = await fetch(`${this.baseUrl}/health`);
        return response.json();
    }
}

// Usage
const api = new FraudDetectionAPI();
const result = await api.predict({
    amount: 15000,
    hour_of_day: 14,
    // ... other fields
});
console.log(`Risk Level: ${result.risk_level}`);
```

---

**API Version**: 1.0  
**Last Updated**: January 2025  
**Contact**: [Your Contact Information]