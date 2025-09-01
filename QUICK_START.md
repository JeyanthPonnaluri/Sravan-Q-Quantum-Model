# 🚀 Quick Start Guide - Neuro-QKAD

## ⚡ 5-Minute Setup

### 1. Install & Train
```bash
# Install dependencies
pip install -r requirements_simple.txt

# Train models (uses all 14 features)
python enhanced_fraud_detector.py
```

### 2. Start API
```bash
# Start server on port 8001
python enhanced_api.py
```

### 3. Test Web Interface
Open: `http://127.0.0.1:8001`

---

## 🎯 Key Features

✅ **14 Transaction Features**: Complete UPI dataset coverage  
✅ **4-Qubit Quantum**: Advanced quantum kernel SVM  
✅ **XGBoost + Rules**: Classical ML + domain logic  
✅ **Fusion Model**: Meta-learning combination  
✅ **Rich Web UI**: All features in one form  

---

## 📊 Example Transaction

```json
{
  "amount": 25000,
  "hour_of_day": 3,
  "is_weekend": 1,
  "day_of_week": "Saturday",
  "sender_age_group": "18-25",
  "receiver_age_group": "26-35",
  "sender_state": "Delhi",
  "sender_bank": "HDFC",
  "receiver_bank": "SBI",
  "merchant_category": "Entertainment",
  "device_type": "Android",
  "transaction_type": "P2P",
  "network_type": "4G",
  "transaction_status": "SUCCESS"
}
```

**Result**: 
- 🔮 Quantum: 0.8%
- 🤖 Classical: 1.3% 
- 🧠 Logic: 45.0%
- ⚡ Fusion: 6.1% → **LOW RISK**

---

## 🔌 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Web interface |
| `/predict` | POST | Fraud analysis |
| `/health` | GET | System status |
| `/info` | GET | Model details |

---

## 📁 Project Files

| File | Purpose |
|------|---------|
| `enhanced_fraud_detector.py` | Training with all features |
| `enhanced_api.py` | FastAPI server + web UI |
| `enhanced_models/` | Saved model artifacts |
| `DOCUMENTATION.md` | Complete documentation |

---

## 🎨 Web Interface Sections

1. **💰 Transaction Details**: Amount, time, weekend
2. **👤 Sender Info**: Age, state, bank
3. **👥 Receiver Info**: Age, bank  
4. **🏪 Transaction Context**: Category, type, status
5. **📱 Technical**: Device, network

---

## 🔧 Troubleshooting

**Models not found?**
```bash
python enhanced_fraud_detector.py
```

**Port in use?**
```bash
# Change port in enhanced_api.py line 671
uvicorn.run(app, host="0.0.0.0", port=8002)
```

**Memory issues?**
- Reduce sample size in training
- Use fewer qubits (2-3)

---

## 🚀 Production Ready

- ✅ Handles unseen categories
- ✅ Proper error handling  
- ✅ Model serialization
- ✅ Responsive web UI
- ✅ Comprehensive logging

**Ready to deploy!** 🎉