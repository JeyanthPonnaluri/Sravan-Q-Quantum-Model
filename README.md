# Neuro-QKAD: Quantum-Classical Fusion Fraud Detection

An end-to-end fraud detection prototype that builds a Practical Fraud Score (0–100) by fusing:
- **Quantum kernel SVM** (PennyLane simulator)
- **Classical XGBoost** baseline
- **Logical score** (LLM / rule-based)

All combined via a fusion meta-model for enhanced fraud detection accuracy.

## 🚀 Quick Start

### Setup & Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training Models

```bash
# Train models (creates models/*.pkl)
python src/train.py --data upi_transactions_2024.csv

# Or with custom parameters
python src/train.py --data upi_transactions_2024.csv --n_qubits 4 --jitter_repeats 20
```

### Running the API

```bash
# Start the FastAPI server
uvicorn src.api_main:app --reload

# Open web interface
# Navigate to: http://127.0.0.1:8000/form
```

### Running Tests

```bash
# Run all tests
pytest -q

# Run specific test files
pytest tests/test_kernel.py -v
pytest tests/test_api.py -v
```

## 📊 Model Architecture

### 1. Data Preprocessing (`src/data_prep.py`)
- Loads CSV data or generates synthetic fallback
- Feature selection and engineering (3-6 configurable features)
- Categorical encoding (one-hot/binary)
- Standard scaling with clipping to [-3, 3]
- Quantum angle conversion: `angle = (scaled/3) * π`
- Time-ordered stratified splits (60% train, 20% cal, 20% test)

### 2. Quantum Kernel (`src/qkernel.py`)
- PennyLane feature map with ≤5 qubits
- Circuit: RY(angle) rotations + shallow CNOT entangling
- Kernel computation: squared fidelity |⟨ψ(a)|ψ(b)⟩|²
- Uncertainty estimation via jittered predictions
- Shot-based evaluation utilities for QPU deployment

### 3. Training Pipeline (`src/train.py`)
- Quantum SVM with precomputed kernels
- Isotonic calibration for probability calibration
- Classical XGBoost baseline
- Rule-based logical scoring
- Fusion meta-model (Logistic Regression)
- Comprehensive evaluation metrics (ROC-AUC, AUPRC, Recall@FPR=0.01, Brier, ECE)

### 4. API Service (`src/api_main.py`)
- FastAPI web service with JSON endpoints
- Real-time fraud scoring
- Uncertainty quantification
- Web UI for interactive testing

## 🎯 API Usage

### Prediction Endpoint

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "amount": 15000,
       "sender_age_group": "26-35",
       "merchant_category": "online",
       "hour_of_day": 2,
       "is_weekend": 1,
       "device_type": "mobile",
       "p_logic": 0.7
     }'
```

### Response Format

```json
{
  "quantum_score": 72.5,
  "classical_score": 65.1,
  "logical_score": 70.0,
  "practical_score": 68.4,
  "uncertainty": 3.2
}
```

## 📈 Features & Configuration

### Supported Features
- `amount`: Transaction amount (INR)
- `sender_age_group`: Age group (18-25, 26-35, 36-50, 50+)
- `merchant_category`: Category (grocery, fuel, restaurant, online, atm)
- `hour_of_day`: Hour (0-23)
- `is_weekend`: Weekend flag (0/1)
- `device_type`: Device (mobile, web, atm)

### Configuration Options

Create `config.yaml` for custom settings:

```yaml
models_dir: 'models'
n_qubits: 4
jitter_std: 0.001
jitter_repeats: 6
uncertainty_enabled: true
```

### Command Line Options

```bash
python src/train.py --help

Options:
  --data TEXT              Path to dataset CSV file
  --features TEXT          Feature columns to use
  --n_qubits INTEGER       Number of qubits (max 5)
  --jitter FLOAT          Jitter std for uncertainty
  --jitter_repeats INTEGER Jittered repeats count
  --save_dir TEXT         Model save directory
```

## 🧪 Testing

The project includes comprehensive tests:

- **Unit Tests** (`tests/test_kernel.py`): Quantum kernel validation
- **Integration Tests** (`tests/test_api.py`): API endpoint testing
- **Validation**: Kernel matrices, value ranges, symmetry properties

## 🔧 Logical Score Integration

### Rule-Based Scoring
The system includes built-in rule-based logical scoring:
- High amount transactions (+30%)
- Late night transactions (+20%)
- Weekend transactions (+10%)
- ATM transactions (+15%)

### LLM Integration
For LLM-based logical scoring, provide `p_logic` parameter (0-1) in API requests:

```python
# Example LLM integration
def get_llm_fraud_score(transaction_context):
    # Call your LLM API here
    prompt = f"Analyze fraud risk for: {transaction_context}"
    # Return score between 0-1
    return llm_api_call(prompt)
```

## 📁 Project Structure

```
neuro_qkad/
├── src/
│   ├── data_prep.py      # Data preprocessing
│   ├── qkernel.py        # Quantum kernel implementation
│   ├── train.py          # Training pipeline
│   ├── api_main.py       # FastAPI service
│   ├── save_load.py      # Model persistence
│   └── web/
│       ├── templates/
│       │   └── index.html
│       └── static/
│           └── script.js
├── tests/
│   ├── test_kernel.py    # Kernel unit tests
│   └── test_api.py       # API integration tests
├── models/               # Trained model artifacts
├── requirements.txt      # Dependencies
├── config.yaml          # Configuration (optional)
└── README.md
```

## 🎛️ Model Artifacts

After training, the following artifacts are saved to `models/`:
- `preprocessor.pkl`: Data preprocessing pipeline
- `quantum_model.pkl`: Calibrated quantum SVM
- `xgboost_model.pkl`: Classical XGBoost model
- `fusion_model.pkl`: Meta-fusion model
- `angles_train.pkl`: Training angles for kernel computation
- `qkernel_config.pkl`: Quantum kernel configuration

## 🚀 Production Considerations

### Performance Optimization
- Use classical prefilter for high-volume scenarios
- Batch quantum kernel computations
- Cache frequently accessed kernels
- Reduce jitter repeats for lower latency

### Monitoring & Auditability
- Log model versions and preprocessing parameters
- Track prediction distributions and drift
- Monitor uncertainty levels
- Implement A/B testing framework

### Scalability
- Deploy with container orchestration
- Use quantum cloud services for larger circuits
- Implement model versioning and rollback
- Add prediction caching layer

## 📊 Evaluation Metrics

The system provides comprehensive evaluation:
- **ROC-AUC**: Area under ROC curve
- **AUPRC**: Area under precision-recall curve
- **Recall@FPR=0.01**: Recall at 1% false positive rate
- **Brier Score**: Probability calibration quality
- **ECE**: Expected calibration error
- **Uncertainty**: Quantum prediction confidence

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PennyLane**: Quantum machine learning framework
- **XGBoost**: Gradient boosting framework
- **FastAPI**: Modern web framework for APIs
- **Scikit-learn**: Machine learning library

---

**Built with ❤️ using Quantum Computing + Classical ML + Modern Web Technologies**