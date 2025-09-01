#!/usr/bin/env python3
"""
Quantum Meta Fraud Detection Web Application
Complete web interface for fraud detection with forms and results
"""

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional
import logging
from datetime import datetime
import uvicorn
import os

# Import our quantum meta model
from quantum_meta_model import QuantumMetaModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Quantum Fraud Detection Web App",
    description="User-friendly web interface for fraud detection",
    version="1.0.0"
)

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global model instance
quantum_model: Optional[QuantumMetaModel] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the quantum model on startup"""
    global quantum_model
    try:
        logger.info("Initializing Quantum Meta Model...")
        quantum_model = QuantumMetaModel()
        logger.info("Quantum Meta Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Quantum Meta Model: {e}")
        raise e

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with transaction form"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_transaction(
    request: Request,
    amount: float = Form(...),
    hour_of_day: int = Form(...),
    is_weekend: int = Form(0),
    day_of_week: str = Form("Monday"),
    sender_age_group: str = Form("26-35"),
    receiver_age_group: str = Form("26-35"),
    sender_state: str = Form("Delhi"),
    sender_bank: str = Form("SBI"),
    receiver_bank: str = Form("SBI"),
    merchant_category: str = Form("Other"),
    device_type: str = Form("Android"),
    transaction_type: str = Form("P2P"),
    network_type: str = Form("4G"),
    transaction_status: str = Form("SUCCESS")
):
    """Analyze transaction and show results"""
    
    if quantum_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Prepare transaction data
        transaction_data = {
            "amount": amount,
            "hour_of_day": hour_of_day,
            "is_weekend": is_weekend,
            "day_of_week": day_of_week,
            "sender_age_group": sender_age_group,
            "receiver_age_group": receiver_age_group,
            "sender_state": sender_state,
            "sender_bank": sender_bank,
            "receiver_bank": receiver_bank,
            "merchant_category": merchant_category,
            "device_type": device_type,
            "transaction_type": transaction_type,
            "network_type": network_type,
            "transaction_status": transaction_status
        }
        
        # Get prediction
        start_time = datetime.now()
        prediction = quantum_model.predict(transaction_data)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Prepare result data
        result_data = {
            "transaction": transaction_data,
            "fraud_score": round(prediction.final_fraud_score, 2),
            "risk_level": prediction.risk_level,
            "confidence": round(prediction.confidence_score, 2),
            "recommended_action": prediction.recommended_action,
            "risk_factors": prediction.primary_risk_factors,
            "fraud_type": prediction.fraud_type_detected,
            "model_agreement": round(prediction.model_agreement, 2),
            "uncertainty": round(prediction.uncertainty_measure, 2),
            "individual_scores": {
                "quantum_score": round(prediction.quantum_score, 2),
                "classical_score": round(prediction.classical_score, 2),
                "neuro_qkad_fusion_score": round(prediction.neuro_qkad_fusion_score, 2),
                "gemini_logical_score": round(prediction.gemini_logical_score, 2)
            },
            "processing_time": round(processing_time, 2),
            "timestamp": prediction.timestamp,
            "model_version": prediction.model_version
        }
        
        return templates.TemplateResponse("results.html", {
            "request": request,
            "result": result_data
        })
        
    except Exception as e:
        logger.error(f"Error analyzing transaction: {e}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if quantum_model is not None else "unhealthy",
        "model_loaded": quantum_model is not None,
        "timestamp": datetime.now().isoformat()
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "quantum_web_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )