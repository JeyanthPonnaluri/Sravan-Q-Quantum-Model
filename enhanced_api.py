"""
Enhanced API for Quantum-Classical Fraud Detection with All Features
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import numpy as np
import pennylane as qml
import os

app = FastAPI(title="Enhanced Quantum-Classical Fraud Detection")

# Global models
models = None


class TransactionFull(BaseModel):
    amount: float
    hour_of_day: int
    is_weekend: int
    day_of_week: str
    sender_age_group: str
    receiver_age_group: str
    sender_state: str
    sender_bank: str
    receiver_bank: str
    merchant_category: str
    device_type: str
    transaction_type: str
    network_type: str
    transaction_status: str


class PredictionResult(BaseModel):
    quantum_score: float
    classical_score: float
    logical_score: float
    fusion_score: float
    risk_level: str
    confidence: str


def quantum_kernel_enhanced(X1, X2, n_qubits=4):
    """Enhanced quantum kernel for prediction"""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        for i in range(min(len(x), n_qubits)):
            qml.RY(x[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.state()

    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            x1 = X1[i][:n_qubits]
            x2 = X2[j][:n_qubits]

            state1 = circuit(x1)
            state2 = circuit(x2)

            K[i, j] = abs(np.vdot(state1, state2)) ** 2

    return K


def compute_logical_scores(X, feature_names):
    """Compute rule-based logical scores"""
    scores = np.zeros(len(X))
    feature_idx = {name: i for i, name in enumerate(feature_names)}

    if "amount (INR)" in feature_idx:
        amount_idx = feature_idx["amount (INR)"]
        high_amounts = X[:, amount_idx] > 10000  # High amount threshold
        scores += 0.3 * high_amounts

    if "hour_of_day" in feature_idx:
        hour_idx = feature_idx["hour_of_day"]
        late_night = (X[:, hour_idx] < 6) | (X[:, hour_idx] > 22)
        scores += 0.2 * late_night

    if "is_weekend" in feature_idx:
        weekend_idx = feature_idx["is_weekend"]
        scores += 0.15 * X[:, weekend_idx]

    if "sender_age_group" in feature_idx:
        age_idx = feature_idx["sender_age_group"]
        young_users = X[:, age_idx] == 0
        scores += 0.1 * young_users

    return np.clip(scores, 0, 1)


def load_models():
    """Load trained models"""
    global models
    try:
        with open("enhanced_models/fraud_models.pkl", "rb") as f:
            models = pickle.load(f)
        print("Enhanced models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False


def predict_fraud_enhanced(transaction: TransactionFull):
    """Predict fraud using all features"""
    if models is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    # Prepare input vector
    X_input = []

    # Numeric features
    X_input.extend(
        [transaction.amount, transaction.hour_of_day, transaction.is_weekend]
    )

    # Categorical features (all non-numeric features)
    categorical_features = [
        ("day_of_week", transaction.day_of_week),
        ("sender_age_group", transaction.sender_age_group),
        ("receiver_age_group", transaction.receiver_age_group),
        ("sender_state", transaction.sender_state),
        ("sender_bank", transaction.sender_bank),
        ("receiver_bank", transaction.receiver_bank),
        ("merchant_category", transaction.merchant_category),
        ("device_type", transaction.device_type),
        ("transaction type", transaction.transaction_type),
        ("network_type", transaction.network_type),
        ("transaction_status", transaction.transaction_status),
    ]

    for feature_name, value in categorical_features:
        if feature_name in models["encoders"]:
            encoder = models["encoders"][feature_name]
            if value in encoder.classes_:
                encoded = encoder.transform([value])[0]
            else:
                encoded = 0  # Default for unseen categories
            X_input.append(encoded)
        else:
            X_input.append(0)

    X_input = np.array(X_input).reshape(1, -1)

    # Scale features
    X_scaled = models["scaler"].transform(X_input)

    # Quantum prediction
    angles_input = (X_scaled[:, : models["n_qubits"]] / 3.0) * np.pi
    K_input = quantum_kernel_enhanced(
        angles_input, models["angles_train"], models["n_qubits"]
    )
    quantum_score = models["quantum_svm"].predict_proba(K_input)[0, 1] * 100

    # Classical prediction
    classical_score = models["xgb_model"].predict_proba(X_scaled)[0, 1] * 100

    # Logical score
    logical_score = compute_logical_scores(X_input, models["feature_names"])[0] * 100

    # Fusion prediction
    fusion_features = np.array(
        [[quantum_score / 100, classical_score / 100, logical_score / 100]]
    )
    fusion_score = models["fusion_model"].predict_proba(fusion_features)[0, 1] * 100

    # Risk level and confidence
    if fusion_score > 70:
        risk_level = "HIGH"
        confidence = "High"
    elif fusion_score > 40:
        risk_level = "MEDIUM"
        confidence = "Medium"
    elif fusion_score > 20:
        risk_level = "LOW-MEDIUM"
        confidence = "Medium"
    else:
        risk_level = "LOW"
        confidence = "High"

    return PredictionResult(
        quantum_score=round(quantum_score, 2),
        classical_score=round(classical_score, 2),
        logical_score=round(logical_score, 2),
        fusion_score=round(fusion_score, 2),
        risk_level=risk_level,
        confidence=confidence,
    )


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    if not load_models():
        print("Warning: Models not loaded. Run enhanced_fraud_detector.py first!")


@app.get("/", response_class=HTMLResponse)
async def home():
    """Enhanced web interface with all features"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced Quantum-Classical Fraud Detection</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            }
            h1 { text-align: center; margin-bottom: 10px; font-size: 2.5em; }
            .subtitle { text-align: center; margin-bottom: 30px; opacity: 0.9; font-size: 1.1em; }
            .form-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                gap: 20px; 
                margin-bottom: 30px;
            }
            .form-group { margin: 15px 0; }
            label { 
                display: block; 
                margin-bottom: 8px; 
                font-weight: bold; 
                font-size: 0.9em;
            }
            input, select { 
                width: 100%; 
                padding: 12px; 
                border: none; 
                border-radius: 8px; 
                font-size: 16px;
                background: rgba(255,255,255,0.9);
                color: #333;
                transition: all 0.3s ease;
            }
            input:focus, select:focus {
                outline: none;
                background: rgba(255,255,255,1);
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }
            button { 
                background: linear-gradient(45deg, #ff6b6b, #ee5a24);
                color: white; 
                padding: 15px 30px; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer;
                font-size: 18px;
                font-weight: bold;
                width: 100%;
                margin-top: 20px;
                transition: all 0.3s ease;
            }
            button:hover { 
                transform: translateY(-3px); 
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            }
            .result { 
                margin-top: 30px; 
                padding: 25px; 
                border-radius: 12px;
                background: rgba(255,255,255,0.15);
                backdrop-filter: blur(5px);
            }
            .scores-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .score-card { 
                padding: 15px;
                background: rgba(255,255,255,0.1);
                border-radius: 8px;
                text-align: center;
                transition: transform 0.3s ease;
            }
            .score-card:hover { transform: translateY(-2px); }
            .score-value { font-size: 1.8em; font-weight: bold; margin: 5px 0; }
            .risk-summary {
                text-align: center;
                padding: 20px;
                margin: 20px 0;
                border-radius: 10px;
                font-size: 1.2em;
                font-weight: bold;
            }
            .high { background: linear-gradient(45deg, #ff4757, #c44569); }
            .medium { background: linear-gradient(45deg, #ffa502, #ff6348); }
            .low-medium { background: linear-gradient(45deg, #3742fa, #2f3542); }
            .low { background: linear-gradient(45deg, #2ed573, #1e90ff); }
            .loading { 
                text-align: center; 
                margin: 20px 0; 
                font-size: 1.1em;
            }
            .feature-section {
                background: rgba(255,255,255,0.05);
                padding: 20px;
                border-radius: 10px;
                margin: 15px 0;
            }
            .section-title {
                font-size: 1.2em;
                font-weight: bold;
                margin-bottom: 15px;
                color: #ffd700;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔮 Enhanced Quantum-Classical Fraud Detection</h1>
            <p class="subtitle">Advanced ML with PennyLane Quantum Computing + XGBoost + Rule-based Logic</p>
            
            <form id="fraudForm">
                <div class="feature-section">
                    <div class="section-title">💰 Transaction Details</div>
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Amount (INR):</label>
                            <input type="number" id="amount" value="15000" step="100" min="1" required>
                        </div>
                        
                        <div class="form-group">
                            <label>Hour of Day (0-23):</label>
                            <input type="number" id="hour_of_day" value="14" min="0" max="23" required>
                        </div>
                        
                        <div class="form-group">
                            <label>Weekend Transaction:</label>
                            <select id="is_weekend">
                                <option value="0">No (Weekday)</option>
                                <option value="1">Yes (Weekend)</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="feature-section">
                    <div class="section-title">👤 Sender Information</div>
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Sender Age Group:</label>
                            <select id="sender_age_group" required>
                                <option value="18-25">18-25 years</option>
                                <option value="26-35" selected>26-35 years</option>
                                <option value="36-45">36-45 years</option>
                                <option value="46-55">46-55 years</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Sender State:</label>
                            <select id="sender_state" required>
                                <option value="Delhi" selected>Delhi</option>
                                <option value="Maharashtra">Maharashtra</option>
                                <option value="Karnataka">Karnataka</option>
                                <option value="Uttar Pradesh">Uttar Pradesh</option>
                                <option value="Tamil Nadu">Tamil Nadu</option>
                                <option value="Gujarat">Gujarat</option>
                                <option value="Rajasthan">Rajasthan</option>
                                <option value="West Bengal">West Bengal</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Sender Bank:</label>
                            <select id="sender_bank" required>
                                <option value="SBI">State Bank of India</option>
                                <option value="HDFC" selected>HDFC Bank</option>
                                <option value="ICICI">ICICI Bank</option>
                                <option value="Axis">Axis Bank</option>
                                <option value="PNB">Punjab National Bank</option>
                                <option value="Kotak">Kotak Mahindra Bank</option>
                                <option value="Yes Bank">Yes Bank</option>
                                <option value="IndusInd">IndusInd Bank</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="feature-section">
                    <div class="section-title">👥 Receiver Information</div>
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Receiver Age Group:</label>
                            <select id="receiver_age_group" required>
                                <option value="18-25">18-25 years</option>
                                <option value="26-35" selected>26-35 years</option>
                                <option value="36-45">36-45 years</option>
                                <option value="46-55">46-55 years</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Receiver Bank:</label>
                            <select id="receiver_bank" required>
                                <option value="SBI" selected>State Bank of India</option>
                                <option value="HDFC">HDFC Bank</option>
                                <option value="ICICI">ICICI Bank</option>
                                <option value="Axis">Axis Bank</option>
                                <option value="PNB">Punjab National Bank</option>
                                <option value="Kotak">Kotak Mahindra Bank</option>
                                <option value="Yes Bank">Yes Bank</option>
                                <option value="IndusInd">IndusInd Bank</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="feature-section">
                    <div class="section-title">🏪 Transaction & Merchant Info</div>
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Day of Week:</label>
                            <select id="day_of_week" required>
                                <option value="Monday">Monday</option>
                                <option value="Tuesday">Tuesday</option>
                                <option value="Wednesday">Wednesday</option>
                                <option value="Thursday">Thursday</option>
                                <option value="Friday" selected>Friday</option>
                                <option value="Saturday">Saturday</option>
                                <option value="Sunday">Sunday</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Merchant Category:</label>
                            <select id="merchant_category" required>
                                <option value="Grocery">Grocery</option>
                                <option value="Fuel">Fuel</option>
                                <option value="Entertainment" selected>Entertainment</option>
                                <option value="Food">Food & Dining</option>
                                <option value="Shopping">Shopping</option>
                                <option value="Utilities">Utilities</option>
                                <option value="Other">Other</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Transaction Type:</label>
                            <select id="transaction_type" required>
                                <option value="P2P" selected>P2P (Person to Person)</option>
                                <option value="P2M">P2M (Person to Merchant)</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Transaction Status:</label>
                            <select id="transaction_status" required>
                                <option value="SUCCESS" selected>SUCCESS</option>
                                <option value="FAILED">FAILED</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="feature-section">
                    <div class="section-title">📱 Technical Details</div>
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Device Type:</label>
                            <select id="device_type" required>
                                <option value="Android" selected>Android</option>
                                <option value="iOS">iOS</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Network Type:</label>
                            <select id="network_type" required>
                                <option value="4G" selected>4G</option>
                                <option value="5G">5G</option>
                                <option value="WiFi">WiFi</option>
                                <option value="3G">3G</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <button type="submit">🔍 Analyze Transaction with Quantum-Classical Fusion</button>
            </form>
            
            <div id="loading" class="loading" style="display: none;">
                <p>🔄 Running quantum-classical analysis with all features...</p>
            </div>
            
            <div id="result" style="display: none;"></div>
        </div>
        
        <script>
            document.getElementById('fraudForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const loading = document.getElementById('loading');
                const result = document.getElementById('result');
                
                loading.style.display = 'block';
                result.style.display = 'none';
                
                const data = {
                    amount: parseFloat(document.getElementById('amount').value),
                    hour_of_day: parseInt(document.getElementById('hour_of_day').value),
                    is_weekend: parseInt(document.getElementById('is_weekend').value),
                    day_of_week: document.getElementById('day_of_week').value,
                    sender_age_group: document.getElementById('sender_age_group').value,
                    receiver_age_group: document.getElementById('receiver_age_group').value,
                    sender_state: document.getElementById('sender_state').value,
                    sender_bank: document.getElementById('sender_bank').value,
                    receiver_bank: document.getElementById('receiver_bank').value,
                    merchant_category: document.getElementById('merchant_category').value,
                    device_type: document.getElementById('device_type').value,
                    transaction_type: document.getElementById('transaction_type').value,
                    network_type: document.getElementById('network_type').value,
                    transaction_status: document.getElementById('transaction_status').value
                };
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    
                    const res = await response.json();
                    
                    const riskClass = res.risk_level.toLowerCase().replace('-', '');
                    
                    result.innerHTML = `
                        <h3>📊 Comprehensive Analysis Results</h3>
                        
                        <div class="risk-summary ${riskClass}">
                            🎯 Risk Level: ${res.risk_level} (${res.confidence} Confidence)
                        </div>
                        
                        <div class="scores-grid">
                            <div class="score-card">
                                <div>🔮 Quantum Score</div>
                                <div class="score-value">${res.quantum_score}%</div>
                                <small>4-qubit quantum kernel SVM</small>
                            </div>
                            
                            <div class="score-card">
                                <div>🤖 Classical Score</div>
                                <div class="score-value">${res.classical_score}%</div>
                                <small>XGBoost with all features</small>
                            </div>
                            
                            <div class="score-card">
                                <div>🧠 Logical Score</div>
                                <div class="score-value">${res.logical_score}%</div>
                                <small>Rule-based analysis</small>
                            </div>
                            
                            <div class="score-card">
                                <div>⚡ Fusion Score</div>
                                <div class="score-value">${res.fusion_score}%</div>
                                <small>Combined final prediction</small>
                            </div>
                        </div>
                        
                        <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 8px;">
                            <strong>Analysis Summary:</strong><br>
                            This transaction was analyzed using quantum computing (${res.quantum_score}%), 
                            classical machine learning (${res.classical_score}%), and rule-based logic (${res.logical_score}%). 
                            The final fusion model predicts a ${res.fusion_score}% fraud probability with ${res.confidence.toLowerCase()} confidence.
                        </div>
                    `;
                    
                    result.style.display = 'block';
                    result.scrollIntoView({ behavior: 'smooth' });
                    
                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    loading.style.display = 'none';
                }
            });
            
            // Set current hour as default
            document.getElementById('hour_of_day').value = new Date().getHours();
            
            // Set weekend based on current day
            const currentDay = new Date().getDay();
            const isWeekend = [0, 6].includes(currentDay) ? 1 : 0;
            document.getElementById('is_weekend').value = isWeekend;
            
            // Set day of week
            const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
            document.getElementById('day_of_week').value = dayNames[currentDay];
        </script>
    </body>
    </html>
    """


@app.post("/predict", response_model=PredictionResult)
async def predict(transaction: TransactionFull):
    """Predict fraud probability using all features"""
    return predict_fraud_enhanced(transaction)


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "models_loaded": models is not None,
        "quantum_enabled": True,
        "features_count": len(models["feature_names"]) if models else 0,
    }


@app.get("/info")
async def info():
    """API information"""
    feature_info = {
        "numeric": ["amount", "hour_of_day", "is_weekend"],
        "categorical": [
            "day_of_week", "sender_age_group", "receiver_age_group", 
            "sender_state", "sender_bank", "receiver_bank", 
            "merchant_category", "device_type", "transaction_type", 
            "network_type", "transaction_status"
        ],
    }

    return {
        "title": "Enhanced Quantum-Classical Fraud Detection",
        "description": "Uses PennyLane quantum kernel SVM + XGBoost + Rule-based logic fusion",
        "features": feature_info,
        "quantum_qubits": models["n_qubits"] if models else 4,
        "models": ["Quantum SVM", "XGBoost", "Logical Rules", "Fusion Model"],
        "total_features": len(models["feature_names"]) if models else 14,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
