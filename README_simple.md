# Simple Quantum-Classical Fraud Detection

A streamlined fraud detection system using quantum kernel SVM + XGBoost fusion.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_simple.txt
```

### 2. Train Models
```bash
python simple_fraud_detector.py
```

### 3. Run API Server
```bash
python simple_api.py
```

### 4. Open Web Interface
Navigate to: http://127.0.0.1:8000

## 📊 What It Does

1. **Quantum Kernel**: Uses PennyLane to create quantum feature maps with 3 qubits
2. **Classical Model**: XGBoost for baseline fraud detection
3. **Fusion**: Combines both models for better accuracy
4. **Web Interface**: Simple form to test transactions

## 🎯 Features Used

- **Amount**: Transaction amount in INR
- **Hour of Day**: When transaction occurred (0-23)
- **Is Weekend**: Whether it's a weekend transaction (0/1)

## 📈 Example Usage

```python
# Test transaction
transaction = {
    'amount': 15000,      # High amount
    'hour_of_day': 2,     # Late night
    'is_weekend': 1       # Weekend
}

# Results might show:
# Quantum Score: 75%
# Classical Score: 68%
# Fusion Score: 71%
# Risk Level: HIGH
```

## 🔧 How It Works

1. **Data Preprocessing**: Scales features and converts to quantum angles
2. **Quantum Circuit**: RY rotations + CNOT gates for feature mapping
3. **Kernel Computation**: Fidelity between quantum states as similarity
4. **Model Training**: SVM on quantum kernel + XGBoost on raw features
5. **Prediction**: Combines both models for final fraud score

## 📁 Files

- `simple_fraud_detector.py` - Main training script
- `simple_api.py` - FastAPI web server
- `requirements_simple.txt` - Dependencies
- `simple_models/` - Saved model artifacts

## ✅ Success Criteria

- ✅ Quantum kernel SVM training
- ✅ Classical XGBoost baseline
- ✅ Model fusion and evaluation
- ✅ Web API with form interface
- ✅ Real-time fraud scoring

Built with ❤️ using PennyLane + XGBoost + FastAPI