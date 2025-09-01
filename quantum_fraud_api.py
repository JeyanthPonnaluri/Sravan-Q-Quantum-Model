#!/usr/bin/env python3
"""
Quantum Meta Model FastAPI Application
Advanced fraud detection API using Quantum + AI fusion
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import asyncio
import uvicorn

# Import our quantum meta model
from quantum_meta_model import QuantumMetaModel, MetaModelPrediction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Quantum Meta Fraud Detection API",
    description="Advanced fraud detection using Quantum + AI fusion technology",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
quantum_model: Optional[QuantumMetaModel] = None

# Pydantic models for request/response
class TransactionRequest(BaseModel):
    """Transaction data for fraud analysis"""
    
    # Required fields
    amount: float = Field(..., description="Transaction amount", ge=0)
    hour_of_day: int = Field(..., description="Hour of day (0-23)", ge=0, le=23)
    
    # Optional fields with defaults
    is_weekend: int = Field(0, description="Is weekend (0 or 1)", ge=0, le=1)
    day_of_week: str = Field("Monday", description="Day of the week")
    sender_age_group: str = Field("26-35", description="Sender age group")
    receiver_age_group: str = Field("26-35", description="Receiver age group")
    sender_state: str = Field("Delhi", description="Sender state")
    sender_bank: str = Field("SBI", description="Sender bank")
    receiver_bank: str = Field("SBI", description="Receiver bank")
    merchant_category: str = Field("Other", description="Merchant category")
    device_type: str = Field("Android", description="Device type")
    transaction_type: str = Field("P2P", description="Transaction type")
    network_type: str = Field("4G", description="Network type")
    transaction_status: str = Field("SUCCESS", description="Transaction status")
    
    @validator('day_of_week')
    def validate_day_of_week(cls, v):
        valid_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        if v not in valid_days:
            raise ValueError(f'Day of week must be one of: {valid_days}')
        return v
    
    @validator('sender_age_group', 'receiver_age_group')
    def validate_age_group(cls, v):
        valid_ages = ['18-25', '26-35', '36-45', '46-55', '56+']
        if v not in valid_ages:
            raise ValueError(f'Age group must be one of: {valid_ages}')
        return v
    
    @validator('device_type')
    def validate_device_type(cls, v):
        valid_devices = ['Android', 'iOS', 'Web', 'Other']
        if v not in valid_devices:
            raise ValueError(f'Device type must be one of: {valid_devices}')
        return v
    
    @validator('transaction_type')
    def validate_transaction_type(cls, v):
        valid_types = ['P2P', 'P2M', 'M2P', 'Other']
        if v not in valid_types:
            raise ValueError(f'Transaction type must be one of: {valid_types}')
        return v
    
    @validator('network_type')
    def validate_network_type(cls, v):
        valid_networks = ['4G', '5G', 'WiFi', '3G', 'Other']
        if v not in valid_networks:
            raise ValueError(f'Network type must be one of: {valid_networks}')
        return v

class FraudAnalysisResponse(BaseModel):
    """Response model for fraud analysis"""
    
    # Core results
    fraud_score: float = Field(..., description="Final fraud score (0-100)")
    risk_level: str = Field(..., description="Risk level classification")
    recommended_action: str = Field(..., description="Recommended action")
    
    # Model scores
    individual_scores: Dict[str, float] = Field(..., description="Individual model scores")
    
    # Analysis details
    confidence_score: float = Field(..., description="Confidence in prediction")
    model_agreement: float = Field(..., description="Agreement between models")
    uncertainty_measure: float = Field(..., description="Uncertainty measure")
    
    # Risk insights
    primary_risk_factors: List[str] = Field(..., description="Primary risk factors identified")
    fraud_type_detected: str = Field(..., description="Type of fraud detected")
    
    # Metadata
    analysis_timestamp: str = Field(..., description="Analysis timestamp")
    model_version: str = Field(..., description="Model version used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class BatchTransactionRequest(BaseModel):
    """Batch transaction analysis request"""
    transactions: List[TransactionRequest] = Field(..., description="List of transactions to analyze")
    
class BatchAnalysisResponse(BaseModel):
    """Batch analysis response"""
    results: List[FraudAnalysisResponse] = Field(..., description="Analysis results for each transaction")
    summary: Dict[str, Any] = Field(..., description="Batch analysis summary")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")

class ModelStatsResponse(BaseModel):
    """Model statistics response"""
    model_version: str
    meta_weights: Dict[str, float]
    risk_thresholds: Dict[str, int]
    neuro_qkad_loaded: bool
    gemini_available: bool
    components: List[str]

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

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if quantum_model is not None else "unhealthy",
        model_loaded=quantum_model is not None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

# Model statistics endpoint
@app.get("/model/stats", response_model=ModelStatsResponse)
async def get_model_stats():
    """Get model statistics"""
    if quantum_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    stats = quantum_model.get_model_statistics()
    return ModelStatsResponse(**stats)

# Main fraud analysis endpoint
@app.post("/analyze", response_model=FraudAnalysisResponse)
async def analyze_transaction(transaction: TransactionRequest):
    """
    Analyze a single transaction for fraud
    
    Returns comprehensive fraud analysis including:
    - Fraud score (0-100)
    - Risk level and recommended action
    - Individual model scores
    - Risk factors and insights
    """
    if quantum_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    start_time = datetime.now()
    
    try:
        # Convert request to dict
        transaction_data = transaction.dict()
        
        # Get prediction from quantum model
        prediction = quantum_model.predict(transaction_data)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Build response
        response = FraudAnalysisResponse(
            fraud_score=round(prediction.final_fraud_score, 2),
            risk_level=prediction.risk_level,
            recommended_action=prediction.recommended_action,
            individual_scores={
                "quantum_score": round(prediction.quantum_score, 2),
                "classical_score": round(prediction.classical_score, 2),
                "neuro_qkad_fusion_score": round(prediction.neuro_qkad_fusion_score, 2),
                "gemini_logical_score": round(prediction.gemini_logical_score, 2)
            },
            confidence_score=round(prediction.confidence_score, 2),
            model_agreement=round(prediction.model_agreement, 2),
            uncertainty_measure=round(prediction.uncertainty_measure, 2),
            primary_risk_factors=prediction.primary_risk_factors,
            fraud_type_detected=prediction.fraud_type_detected,
            analysis_timestamp=prediction.timestamp,
            model_version=prediction.model_version,
            processing_time_ms=round(processing_time, 2)
        )
        
        logger.info(f"Transaction analyzed: Amount=₹{transaction.amount:,}, Score={prediction.final_fraud_score:.1f}%, Risk={prediction.risk_level}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing transaction: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Batch analysis endpoint
@app.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch_transactions(batch_request: BatchTransactionRequest):
    """
    Analyze multiple transactions in batch
    
    Processes multiple transactions and returns analysis for each
    along with batch summary statistics
    """
    if quantum_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    if len(batch_request.transactions) > 100:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100 transactions")
    
    start_time = datetime.now()
    results = []
    
    try:
        # Process each transaction
        for i, transaction in enumerate(batch_request.transactions):
            try:
                transaction_data = transaction.dict()
                prediction = quantum_model.predict(transaction_data)
                
                result = FraudAnalysisResponse(
                    fraud_score=round(prediction.final_fraud_score, 2),
                    risk_level=prediction.risk_level,
                    recommended_action=prediction.recommended_action,
                    individual_scores={
                        "quantum_score": round(prediction.quantum_score, 2),
                        "classical_score": round(prediction.classical_score, 2),
                        "neuro_qkad_fusion_score": round(prediction.neuro_qkad_fusion_score, 2),
                        "gemini_logical_score": round(prediction.gemini_logical_score, 2)
                    },
                    confidence_score=round(prediction.confidence_score, 2),
                    model_agreement=round(prediction.model_agreement, 2),
                    uncertainty_measure=round(prediction.uncertainty_measure, 2),
                    primary_risk_factors=prediction.primary_risk_factors,
                    fraud_type_detected=prediction.fraud_type_detected,
                    analysis_timestamp=prediction.timestamp,
                    model_version=prediction.model_version,
                    processing_time_ms=0  # Will be calculated for batch
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing transaction {i}: {e}")
                # Continue with other transactions
                continue
        
        # Calculate batch statistics
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        if results:
            fraud_scores = [r.fraud_score for r in results]
            risk_levels = [r.risk_level for r in results]
            
            summary = {
                "total_transactions": len(batch_request.transactions),
                "successful_analyses": len(results),
                "failed_analyses": len(batch_request.transactions) - len(results),
                "average_fraud_score": round(sum(fraud_scores) / len(fraud_scores), 2),
                "high_risk_count": sum(1 for level in risk_levels if level in ["HIGH", "CRITICAL"]),
                "processing_time_ms": round(processing_time, 2),
                "timestamp": datetime.now().isoformat()
            }
        else:
            summary = {
                "total_transactions": len(batch_request.transactions),
                "successful_analyses": 0,
                "failed_analyses": len(batch_request.transactions),
                "error": "All transactions failed to process"
            }
        
        return BatchAnalysisResponse(results=results, summary=summary)
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

# Quick score endpoint (simplified)
@app.post("/score")
async def get_fraud_score(transaction: TransactionRequest):
    """
    Get just the fraud score (simplified endpoint)
    
    Returns only the fraud score as a number for quick integration
    """
    if quantum_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        transaction_data = transaction.dict()
        prediction = quantum_model.predict(transaction_data)
        
        return {
            "fraud_score": round(prediction.final_fraud_score, 2),
            "risk_level": prediction.risk_level,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting fraud score: {e}")
        raise HTTPException(status_code=500, detail=f"Score calculation failed: {str(e)}")

# Example transaction endpoint
@app.get("/example")
async def get_example_transaction():
    """Get an example transaction for testing"""
    return {
        "example_transaction": {
            "amount": 85000,
            "hour_of_day": 2,
            "is_weekend": 1,
            "day_of_week": "Saturday",
            "sender_age_group": "18-25",
            "receiver_age_group": "46-55",
            "sender_state": "Delhi",
            "sender_bank": "HDFC",
            "receiver_bank": "Unknown Bank",
            "merchant_category": "Entertainment",
            "device_type": "Android",
            "transaction_type": "P2P",
            "network_type": "WiFi",
            "transaction_status": "SUCCESS"
        },
        "usage": "POST this data to /analyze endpoint"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Quantum Meta Fraud Detection API",
        "version": "1.0.0",
        "description": "Advanced fraud detection using Quantum + AI fusion technology",
        "endpoints": {
            "analyze": "POST /analyze - Analyze single transaction",
            "batch": "POST /analyze/batch - Analyze multiple transactions",
            "score": "POST /score - Get fraud score only",
            "health": "GET /health - Health check",
            "stats": "GET /model/stats - Model statistics",
            "example": "GET /example - Example transaction",
            "docs": "GET /docs - API documentation"
        }
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "quantum_fraud_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )