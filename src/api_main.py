"""
FastAPI application for quantum-classical fusion fraud detection.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import os
import sys
import yaml
from contextlib import asynccontextmanager

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from save_load import load_models, load
from qkernel import QuantumKernel


# Global variables for loaded models
models = {}
preprocessor = None
qkernel = None
config = {}


class TransactionInput(BaseModel):
    """Input schema for fraud prediction."""
    amount: float
    sender_age_group: str = "26-35"
    merchant_category: str = "Grocery"
    hour_of_day: int = 12
    is_weekend: int = 0
    device_type: str = "Android"
    p_logic: Optional[float] = None


class PredictionOutput(BaseModel):
    """Output schema for fraud prediction."""
    quantum_score: float
    classical_score: float
    logical_score: float
    practical_score: float
    uncertainty: float


def load_config():
    """Load configuration from config.yaml or use defaults."""
    config_path = "config.yaml"
    default_config = {
        'models_dir': 'models',
        'n_qubits': 4,
        'jitter_std': 0.001,
        'jitter_repeats': 6,  # Reduced for low-latency demo
        'uncertainty_enabled': True
    }
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        # Merge with defaults
        for key, value in default_config.items():
            if key not in loaded_config:
                loaded_config[key] = value
        return loaded_config
    else:
        return default_config


def load_all_models():
    """Load all trained models and preprocessor."""
    global models, preprocessor, qkernel, config
    
    config = load_config()
    models_dir = config['models_dir']
    
    try:
        # Load models
        model_names = ['preprocessor', 'quantum_model', 'xgboost_model', 'fusion_model', 'angles_train']
        models = load_models(model_names, models_dir)
        
        # Set preprocessor
        preprocessor = models['preprocessor']
        
        # Initialize quantum kernel
        qkernel_config_path = os.path.join(models_dir, 'qkernel_config.pkl')
        if os.path.exists(qkernel_config_path):
            qkernel_config = load(qkernel_config_path)
            n_qubits = qkernel_config.get('n_qubits', config['n_qubits'])
        else:
            n_qubits = config['n_qubits']
        
        qkernel = QuantumKernel(n_qubits=n_qubits)
        
        print(f"Successfully loaded models from {models_dir}")
        print(f"Quantum kernel initialized with {n_qubits} qubits")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    load_all_models()
    yield


# Initialize FastAPI app
app = FastAPI(
    title="Quantum-Classical Fraud Detection API",
    description="End-to-end fraud detection using quantum kernel SVM, XGBoost, and fusion modeling",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files and templates
os.makedirs("src/web/static", exist_ok=True)
os.makedirs("src/web/templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
templates = Jinja2Templates(directory="src/web/templates")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to form."""
    return """
    <html>
        <head><title>Fraud Detection API</title></head>
        <body>
            <h1>Quantum-Classical Fraud Detection</h1>
            <p><a href="/form">Go to Prediction Form</a></p>
            <p><a href="/docs">API Documentation</a></p>
        </body>
    </html>
    """


@app.get("/form", response_class=HTMLResponse)
async def get_form(request: Request):
    """Serve the prediction form."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_model=PredictionOutput)
async def predict_fraud(transaction: TransactionInput):
    """
    Predict fraud probability for a transaction.
    
    Returns scores scaled to 0-100 range with uncertainty estimation.
    """
    try:
        # Validate models are loaded
        if not models or preprocessor is None or qkernel is None:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Prepare input data
        input_data = pd.DataFrame([{
            'amount (INR)': transaction.amount,
            'sender_age_group': transaction.sender_age_group,
            'merchant_category': transaction.merchant_category,
            'hour_of_day': transaction.hour_of_day,
            'is_weekend': transaction.is_weekend,
            'device_type': transaction.device_type
        }])
        
        # Preprocess input
        X_processed, angles_input = preprocessor.preprocess_features(input_data, fit=False)
        
        # Get quantum prediction
        angles_train = models['angles_train']
        K_input = qkernel.kernel_matrix(angles_input, angles_train)
        p_quantum = models['quantum_model'].predict_proba(K_input)[0, 1]
        
        # Get classical prediction
        p_classical = models['xgboost_model'].predict_proba(X_processed)[0, 1]
        
        # Get logical score
        p_logic = transaction.p_logic if transaction.p_logic is not None else 0.0
        p_logic = max(0.0, min(1.0, p_logic))  # Clip to [0, 1]
        
        # Fusion prediction
        fusion_features = np.array([[p_quantum, p_classical, p_logic]])
        p_fusion = models['fusion_model'].predict_proba(fusion_features)[0, 1]
        
        # Compute uncertainty if enabled
        uncertainty = 0.0
        if config.get('uncertainty_enabled', True):
            try:
                _, uncertainty = qkernel.compute_uncertainty(
                    angles_input[0], angles_train, models['quantum_model'],
                    n_repeats=config['jitter_repeats'], 
                    jitter_std=config['jitter_std']
                )
                uncertainty = float(uncertainty * 100)  # Scale to 0-100
            except Exception as e:
                print(f"Warning: Could not compute uncertainty: {e}")
                uncertainty = 0.0
        
        # Scale scores to 0-100 and round
        quantum_score = round(float(p_quantum * 100), 2)
        classical_score = round(float(p_classical * 100), 2)
        logical_score = round(float(p_logic * 100), 2)
        practical_score = round(float(p_fusion * 100), 2)
        uncertainty = round(uncertainty, 2)
        
        return PredictionOutput(
            quantum_score=quantum_score,
            classical_score=classical_score,
            logical_score=logical_score,
            practical_score=practical_score,
            uncertainty=uncertainty
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(models) > 0,
        "quantum_kernel_ready": qkernel is not None
    }


@app.get("/model_info")
async def model_info():
    """Get information about loaded models."""
    if not models:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    return {
        "loaded_models": list(models.keys()),
        "feature_columns": preprocessor.feature_columns if preprocessor else [],
        "n_qubits": qkernel.n_qubits if qkernel else 0,
        "config": config
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)