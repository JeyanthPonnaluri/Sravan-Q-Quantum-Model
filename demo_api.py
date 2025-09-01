"""
Demo API for Quantum-Classical Fraud Detection
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import numpy as np
import pennylane as qml
import os

app = FastAPI(title="Quantum-Classical Fraud Detection Demo")

# Global models
models = None

class Transaction(BaseModel):
    amount: float
    hour_of_day: int
    is_weekend: int

class PredictionResult(BaseModel):
    quantum_score: float
    classical_score: float
    fusion_score: float
    risk_level: str

def quantum_kernel_simple(X1, X2):
    """Simple quantum kernel for prediction"""
    dev = qml.device('default.qubit', wires=2)
    
    @qml.qnode(dev)
    def circuit(x):
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.state()
    
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            x1 = X1[i][:2]
            x2 = X2[j][:2]
            
            state1 = circuit(x1)
            state2 = circuit(x2)
            
            K[i, j] = abs(np.vdot(state1, state2))**2
    
    return K

def load_models():
    """Load trained models"""
    global models
    try:
        with open('demo_models/fraud_models.pkl', 'rb') as f:
            models = pickle.load(f)
        print("Models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def predict_fraud(transaction: Transaction):
    """Predict fraud for a transaction"""
    if models is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    # Prepare input
    X_input = np.array([[transaction.amount, transaction.hour_of_day, transaction.is_weekend]])
    
    # Scale features
    X_scaled = models['scaler'].transform(X_input)
    
    # Convert to quantum angles (first 2 features)
    angles_input = (X_scaled[:, :2] / 3.0) * np.pi
    
    # Quantum prediction
    K_input = quantum_kernel_simple(angles_input, models['angles_train'])
    quantum_score = models['quantum_svm'].predict_proba(K_input)[0, 1] * 100
    
    # Classical prediction
    classical_score = models['xgb_model'].predict_proba(X_scaled)[0, 1] * 100
    
    # Fusion score
    fusion_score = (quantum_score + classical_score) / 2
    
    # Risk level
    if fusion_score > 50:
        risk_level = "HIGH"
    elif fusion_score > 20:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return PredictionResult(
        quantum_score=round(quantum_score, 2),
        classical_score=round(classical_score, 2),
        fusion_score=round(fusion_score, 2),
        risk_level=risk_level
    )

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    if not load_models():
        print("Warning: Models not loaded. Run minimal_fraud_detector.py first!")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Demo web interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Quantum-Classical Fraud Detection Demo</title>
        <style>
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                max-width: 800px; 
                margin: 50px auto; 
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            }
            h1 { text-align: center; margin-bottom: 10px; }
            .subtitle { text-align: center; margin-bottom: 30px; opacity: 0.8; }
            .form-group { margin: 20px 0; }
            label { display: block; margin-bottom: 8px; font-weight: bold; }
            input, select { 
                width: 100%; 
                padding: 12px; 
                border: none; 
                border-radius: 8px; 
                font-size: 16px;
                background: rgba(255,255,255,0.9);
                color: #333;
            }
            button { 
                background: linear-gradient(45deg, #ff6b6b, #ee5a24);
                color: white; 
                padding: 15px 30px; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                width: 100%;
                margin-top: 20px;
                transition: transform 0.2s;
            }
            button:hover { transform: translateY(-2px); }
            .result { 
                margin-top: 30px; 
                padding: 20px; 
                border-radius: 10px;
                background: rgba(255,255,255,0.15);
            }
            .score { 
                display: flex; 
                justify-content: space-between; 
                margin: 10px 0;
                padding: 10px;
                background: rgba(255,255,255,0.1);
                border-radius: 5px;
            }
            .high { border-left: 5px solid #ff4757; }
            .medium { border-left: 5px solid #ffa502; }
            .low { border-left: 5px solid #2ed573; }
            .loading { text-align: center; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔮 Quantum-Classical Fraud Detection</h1>
            <p class="subtitle">Advanced ML using PennyLane Quantum Computing + XGBoost</p>
            
            <form id="fraudForm">
                <div class="form-group">
                    <label>💰 Transaction Amount (INR):</label>
                    <input type="number" id="amount" value="5000" step="100" min="1" required>
                </div>
                
                <div class="form-group">
                    <label>🕐 Hour of Day (0-23):</label>
                    <input type="number" id="hour_of_day" value="14" min="0" max="23" required>
                </div>
                
                <div class="form-group">
                    <label>📅 Weekend Transaction:</label>
                    <select id="is_weekend">
                        <option value="0">No (Weekday)</option>
                        <option value="1">Yes (Weekend)</option>
                    </select>
                </div>
                
                <button type="submit">🔍 Analyze Transaction</button>
            </form>
            
            <div id="loading" class="loading" style="display: none;">
                <p>🔄 Running quantum-classical analysis...</p>
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
                    is_weekend: parseInt(document.getElementById('is_weekend').value)
                };
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    
                    const res = await response.json();
                    
                    const riskClass = res.risk_level.toLowerCase();
                    
                    result.innerHTML = `
                        <h3>📊 Analysis Results</h3>
                        <div class="score">
                            <span>🔮 Quantum Score:</span>
                            <strong>${res.quantum_score}%</strong>
                        </div>
                        <div class="score">
                            <span>🤖 Classical Score:</span>
                            <strong>${res.classical_score}%</strong>
                        </div>
                        <div class="score">
                            <span>⚡ Fusion Score:</span>
                            <strong>${res.fusion_score}%</strong>
                        </div>
                        <div class="score ${riskClass}">
                            <span>🎯 Risk Level:</span>
                            <strong>${res.risk_level}</strong>
                        </div>
                    `;
                    
                    result.style.display = 'block';
                    
                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    loading.style.display = 'none';
                }
            });
            
            // Set current hour as default
            document.getElementById('hour_of_day').value = new Date().getHours();
        </script>
    </body>
    </html>
    """

@app.post("/predict", response_model=PredictionResult)
async def predict(transaction: Transaction):
    """Predict fraud probability"""
    return predict_fraud(transaction)

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy", 
        "models_loaded": models is not None,
        "quantum_enabled": True
    }

@app.get("/info")
async def info():
    """API information"""
    return {
        "title": "Quantum-Classical Fraud Detection Demo",
        "description": "Uses PennyLane quantum kernel SVM + XGBoost fusion",
        "features": ["amount", "hour_of_day", "is_weekend"],
        "quantum_qubits": 2,
        "models": ["Quantum SVM", "XGBoost", "Fusion"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)