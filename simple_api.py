"""
Simple FastAPI server for fraud detection
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Simple Fraud Detection API")

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

def load_models():
    """Load trained models"""
    global models
    try:
        models = joblib.load('simple_models/fraud_models.pkl')
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        models = None

def predict_fraud(transaction_data):
    """Predict fraud for a transaction"""
    if models is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    # Prepare input
    X_input = np.array([[
        transaction_data.amount,
        transaction_data.hour_of_day,
        transaction_data.is_weekend
    ]])
    
    # Scale and convert to angles
    X_scaled = models['scaler'].transform(X_input)
    angles_input = (X_scaled / 3.0) * np.pi
    
    # Quantum prediction
    K_input = models['qkernel'].compute_kernel(angles_input, models['angles_train'])
    quantum_score = models['quantum_svm'].predict_proba(K_input)[0, 1] * 100
    
    # Classical prediction
    classical_score = models['xgb_model'].predict_proba(X_scaled)[0, 1] * 100
    
    # Fusion score
    fusion_score = (quantum_score + classical_score) / 2
    
    # Risk level
    if fusion_score > 70:
        risk_level = "HIGH"
    elif fusion_score > 40:
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
    load_models()

@app.get("/", response_class=HTMLResponse)
async def home():
    """Simple home page with form"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simple Fraud Detection</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
            .form-group { margin: 15px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { margin-top: 20px; padding: 15px; border-radius: 4px; }
            .high { background: #f8d7da; border: 1px solid #f5c6cb; }
            .medium { background: #fff3cd; border: 1px solid #ffeaa7; }
            .low { background: #d4edda; border: 1px solid #c3e6cb; }
        </style>
    </head>
    <body>
        <h1>🔮 Simple Fraud Detection</h1>
        <p>Quantum + Classical ML for fraud detection</p>
        
        <form id="fraudForm">
            <div class="form-group">
                <label>Amount (INR):</label>
                <input type="number" id="amount" value="5000" required>
            </div>
            
            <div class="form-group">
                <label>Hour of Day (0-23):</label>
                <input type="number" id="hour_of_day" value="14" min="0" max="23" required>
            </div>
            
            <div class="form-group">
                <label>Weekend Transaction:</label>
                <select id="is_weekend">
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>
            
            <button type="submit">Analyze Transaction</button>
        </form>
        
        <div id="result" style="display: none;"></div>
        
        <script>
            document.getElementById('fraudForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
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
                    
                    const result = await response.json();
                    
                    const resultDiv = document.getElementById('result');
                    const riskClass = result.risk_level.toLowerCase();
                    
                    resultDiv.innerHTML = `
                        <h3>Analysis Results</h3>
                        <p><strong>Quantum Score:</strong> ${result.quantum_score}%</p>
                        <p><strong>Classical Score:</strong> ${result.classical_score}%</p>
                        <p><strong>Fusion Score:</strong> ${result.fusion_score}%</p>
                        <p><strong>Risk Level:</strong> ${result.risk_level}</p>
                    `;
                    resultDiv.className = `result ${riskClass}`;
                    resultDiv.style.display = 'block';
                    
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            });
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
    return {"status": "healthy", "models_loaded": models is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)