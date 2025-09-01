# Enhanced Quantum-Classical Fraud Detection

A comprehensive fraud detection system using **all available features** from UPI transaction data, combining quantum computing, classical ML, and rule-based logic.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_simple.txt
```

### 2. Train Enhanced Models
```bash
python enhanced_fraud_detector.py
```

### 3. Run Enhanced API
```bash
python enhanced_api.py
```

### 4. Open Web Interface
Navigate to: **http://127.0.0.1:8001**

## 🎯 All Features Used

### 💰 Transaction Details
- **Amount (INR)**: Transaction value
- **Hour of Day**: Time when transaction occurred (0-23)
- **Is Weekend**: Weekend vs weekday transaction

### 👤 User & Merchant Info
- **Sender Age Group**: 18-25, 26-35, 36-45, 46-55
- **Merchant Category**: Grocery, Fuel, Entertainment, Food, Shopping, Utilities, Other
- **Transaction Type**: P2P (Person to Person) or P2M (Person to Merchant)

### 📱 Technical Details
- **Device Type**: Android or iOS
- **Network Type**: 4G, 5G, WiFi, 3G

## 🔮 Model Architecture

### 1. **Quantum Kernel SVM** (4 qubits)
- Uses first 4 features in quantum circuit
- RY rotations + CNOT entangling gates
- Fidelity-based kernel computation

### 2. **Classical XGBoost** (all 8 features)
- Gradient boosting with all categorical and numeric features
- Handles class imbalance with scale_pos_weight
- 50 estimators, max_depth=4

### 3. **Rule-based Logic**
- High amount transactions (>10k INR): +30%
- Late night/early morning (22-6h): +20%
- Weekend transactions: +15%
- Young users (18-25): +10%

### 4. **Fusion Model**
- Logistic regression combining all three scores
- Balanced class weights
- Final fraud probability 0-100%

## 📊 Example Analysis

```json
{
  "amount": 25000,
  "hour_of_day": 3,
  "is_weekend": 1,
  "sender_age_group": "18-25",
  "merchant_category": "Entertainment",
  "device_type": "Android",
  "transaction_type": "P2P",
  "network_type": "4G"
}
```

**Results:**
- 🔮 Quantum Score: 0.8%
- 🤖 Classical Score: 2.4%
- 🧠 Logical Score: 45.0%
- ⚡ Fusion Score: 6.4%
- 🎯 Risk Level: LOW

## 🎨 Web Interface Features

- **Comprehensive Form**: All 8 transaction features
- **Real-time Analysis**: Quantum + Classical + Logic fusion
- **Visual Results**: Color-coded risk levels and score cards
- **Responsive Design**: Works on desktop and mobile
- **Smart Defaults**: Auto-sets current time and weekend status

## 🔧 API Endpoints

### POST `/predict`
Analyze transaction with all features:
```bash
curl -X POST "http://127.0.0.1:8001/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "amount": 15000,
       "hour_of_day": 14,
       "is_weekend": 0,
       "sender_age_group": "26-35",
       "merchant_category": "Grocery",
       "device_type": "Android",
       "transaction_type": "P2M",
       "network_type": "4G"
     }'
```

### GET `/info`
Get model and feature information

### GET `/health`
Health check and model status

## 🎯 Key Improvements

✅ **All 8 Features**: Uses complete UPI transaction feature set  
✅ **4-Qubit Quantum**: Enhanced quantum circuit with more qubits  
✅ **Advanced Fusion**: Logistic regression fusion model  
✅ **Rule-based Logic**: Domain-specific fraud detection rules  
✅ **Rich Web UI**: Comprehensive form with all features  
✅ **Better Encoding**: Proper categorical feature handling  
✅ **Class Balancing**: Handles imbalanced fraud datasets  

## 📁 Files

- `enhanced_fraud_detector.py` - Training with all features
- `enhanced_api.py` - FastAPI server with comprehensive UI
- `enhanced_models/` - Saved model artifacts
- `README_enhanced.md` - This documentation

## 🚀 Production Ready

- Handles unseen categorical values gracefully
- Proper feature scaling and encoding
- Model serialization with all components
- Comprehensive error handling
- Responsive web interface

**Built with ❤️ using PennyLane Quantum Computing + XGBoost + FastAPI + All UPI Features**