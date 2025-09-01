"""
Meta Fraud Detection API
Complete FastAPI application integrating:
1. Neuro-QKAD (Quantum + Classical + Rules)
2. Gemini AI Logical Model
3. Quantum Meta Model Fusion
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
import logging
import asyncio
from datetime import datetime
import json

# Import our models
from quantum_meta_model import QuantumMetaModel, MetaModelPrediction
from gemini_logical_model import GeminiLogicalModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Meta Fraud Detection API",
    description="Advanced fraud detection using Quantum Computing + AI + Machine Learning",
    version="2.0.0"
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
meta_model: Optional[QuantumMetaModel] = None

class TransactionInput(BaseModel):
    """Complete transaction input model with validation"""
    amount: float = Field(..., gt=0, le=10000000, description="Transaction amount in INR")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    is_weekend: int = Field(..., ge=0, le=1, description="Weekend flag (0/1)")
    day_of_week: str = Field(..., description="Day of the week")
    sender_age_group: str = Field(..., description="Sender age group")
    receiver_age_group: str = Field(..., description="Receiver age group")
    sender_state: str = Field(..., description="Sender state")
    sender_bank: str = Field(..., description="Sender bank")
    receiver_bank: str = Field(..., description="Receiver bank")
    merchant_category: str = Field(..., description="Merchant category")
    device_type: str = Field(..., description="Device type")
    transaction_type: str = Field(..., description="Transaction type")
    network_type: str = Field(..., description="Network type")
    transaction_status: str = Field(..., description="Transaction status")
    
    @validator('day_of_week')
    def validate_day_of_week(cls, v):
        valid_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        if v not in valid_days:
            raise ValueError(f'Day must be one of {valid_days}')
        return v
    
    @validator('sender_age_group', 'receiver_age_group')
    def validate_age_groups(cls, v):
        valid_ages = ['18-25', '26-35', '36-45', '46-55']
        if v not in valid_ages:
            raise ValueError(f'Age group must be one of {valid_ages}')
        return v
    
    @validator('transaction_type')
    def validate_transaction_type(cls, v):
        valid_types = ['P2P', 'P2M']
        if v not in valid_types:
            raise ValueError(f'Transaction type must be one of {valid_types}')
        return v
    
    @validator('device_type')
    def validate_device_type(cls, v):
        valid_devices = ['Android', 'iOS']
        if v not in valid_devices:
            raise ValueError(f'Device type must be one of {valid_devices}')
        return v
    
    @validator('network_type')
    def validate_network_type(cls, v):
        valid_networks = ['4G', '5G', 'WiFi', '3G']
        if v not in valid_networks:
            raise ValueError(f'Network type must be one of {valid_networks}')
        return v
    
    @validator('transaction_status')
    def validate_transaction_status(cls, v):
        valid_statuses = ['SUCCESS', 'FAILED']
        if v not in valid_statuses:
            raise ValueError(f'Transaction status must be one of {valid_statuses}')
        return v

class MetaPredictionResponse(BaseModel):
    """Complete meta model prediction response"""
    # Individual model scores
    quantum_score: float
    classical_score: float
    neuro_qkad_fusion_score: float
    gemini_logical_score: float
    
    # Meta model results
    final_fraud_score: float
    confidence_score: float
    risk_level: str
    
    # Analysis insights
    primary_risk_factors: List[str]
    fraud_type_detected: str
    recommended_action: str
    model_agreement: float
    uncertainty_measure: float
    
    # Metadata
    timestamp: str
    model_version: str
    processing_time_ms: float

def load_meta_model():
    """Load the meta model on startup"""
    global meta_model
    try:
        meta_model = QuantumMetaModel()
        logger.info("Meta model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load meta model: {e}")
        meta_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting Meta Fraud Detection API...")
    load_meta_model()

@app.get("/", response_class=HTMLResponse)
async def home():
    """Comprehensive web interface for meta fraud detection"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Meta Fraud Detection - Quantum AI System</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(15px);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            }
            
            .header {
                text-align: center;
                margin-bottom: 40px;
            }
            
            .header h1 {
                font-size: 3em;
                margin-bottom: 10px;
                background: linear-gradient(45deg, #FFD700, #FFA500);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .header .subtitle {
                font-size: 1.2em;
                opacity: 0.9;
                margin-bottom: 20px;
            }
            
            .model-badges {
                display: flex;
                justify-content: center;
                gap: 15px;
                flex-wrap: wrap;
                margin-bottom: 30px;
            }
            
            .badge {
                background: rgba(255,255,255,0.2);
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 0.9em;
                border: 1px solid rgba(255,255,255,0.3);
            }
            
            .form-container {
                background: rgba(255,255,255,0.05);
                border-radius: 15px;
                padding: 30px;
                margin-bottom: 30px;
            }
            
            .form-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 25px;
                margin-bottom: 30px;
            }
            
            .form-section {
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 12px;
                border-left: 4px solid #FFD700;
            }
            
            .section-title {
                font-size: 1.3em;
                font-weight: bold;
                margin-bottom: 15px;
                color: #FFD700;
            }
            
            .form-group {
                margin-bottom: 15px;
            }
            
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: 600;
                font-size: 0.95em;
            }
            
            input, select {
                width: 100%;
                padding: 12px;
                border: none;
                border-radius: 8px;
                background: rgba(255,255,255,0.9);
                color: #333;
                font-size: 16px;
                transition: all 0.3s ease;
            }
            
            input:focus, select:focus {
                outline: none;
                background: rgba(255,255,255,1);
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            
            .analyze-btn {
                background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
                color: white;
                padding: 18px 40px;
                border: none;
                border-radius: 12px;
                font-size: 18px;
                font-weight: bold;
                cursor: pointer;
                width: 100%;
                margin-top: 20px;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .analyze-btn:hover {
                transform: translateY(-3px);
                box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            }
            
            .analyze-btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .loading {
                display: none;
                text-align: center;
                margin: 30px 0;
            }
            
            .loading-spinner {
                width: 50px;
                height: 50px;
                border: 4px solid rgba(255,255,255,0.3);
                border-top: 4px solid #FFD700;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .results {
                display: none;
                background: rgba(255,255,255,0.1);
                border-radius: 15px;
                padding: 30px;
                margin-top: 30px;
            }
            
            .results-header {
                text-align: center;
                margin-bottom: 30px;
            }
            
            .results-header h2 {
                font-size: 2.2em;
                margin-bottom: 10px;
            }
            
            .final-score {
                text-align: center;
                margin: 30px 0;
                padding: 25px;
                border-radius: 15px;
                font-size: 1.5em;
            }
            
            .score-critical { background: linear-gradient(45deg, #FF4757, #C44569); }
            .score-high { background: linear-gradient(45deg, #FF6348, #FF4757); }
            .score-medium { background: linear-gradient(45deg, #FFA502, #FF6348); }
            .score-low { background: linear-gradient(45deg, #26de81, #20bf6b); }
            .score-minimal { background: linear-gradient(45deg, #0abde3, #006ba6); }
            
            .scores-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            
            .score-card {
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                transition: transform 0.3s ease;
            }
            
            .score-card:hover {
                transform: translateY(-5px);
            }
            
            .score-card .score-label {
                font-size: 0.9em;
                opacity: 0.8;
                margin-bottom: 8px;
            }
            
            .score-card .score-value {
                font-size: 2em;
                font-weight: bold;
                margin-bottom: 5px;
            }
            
            .score-card .score-desc {
                font-size: 0.8em;
                opacity: 0.7;
            }
            
            .insights-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }
            
            .insight-card {
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 12px;
            }
            
            .insight-title {
                font-weight: bold;
                margin-bottom: 10px;
                color: #FFD700;
            }
            
            .risk-factors {
                list-style: none;
                padding: 0;
            }
            
            .risk-factors li {
                background: rgba(255,255,255,0.1);
                padding: 8px 12px;
                margin: 5px 0;
                border-radius: 6px;
                border-left: 3px solid #FF6B6B;
            }
            
            .recommended-action {
                background: rgba(255,255,255,0.15);
                padding: 15px;
                border-radius: 10px;
                margin-top: 20px;
                border-left: 4px solid #4ECDC4;
            }
            
            @media (max-width: 768px) {
                .container { padding: 20px; }
                .header h1 { font-size: 2em; }
                .form-grid { grid-template-columns: 1fr; }
                .scores-grid { grid-template-columns: repeat(2, 1fr); }
                .insights-grid { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔮 Meta Fraud Detection</h1>
                <p class="subtitle">Advanced AI-Powered Transaction Analysis</p>
                <div class="model-badges">
                    <span class="badge">🔮 Quantum Computing</span>
                    <span class="badge">🤖 Machine Learning</span>
                    <span class="badge">🧠 Gemini AI</span>
                    <span class="badge">⚡ Meta Fusion</span>
                </div>
            </div>
            
            <div class="form-container">
                <form id="fraudForm">
                    <div class="form-grid">
                        <div class="form-section">
                            <div class="section-title">💰 Transaction Details</div>
                            <div class="form-group">
                                <label>Amount (INR):</label>
                                <input type="number" id="amount" value="25000" step="100" min="1" max="10000000" required>
                            </div>
                            <div class="form-group">
                                <label>Hour of Day (0-23):</label>
                                <input type="number" id="hour_of_day" value="14" min="0" max="23" required>
                            </div>
                            <div class="form-group">
                                <label>Weekend Transaction:</label>
                                <select id="is_weekend" required>
                                    <option value="0">No (Weekday)</option>
                                    <option value="1">Yes (Weekend)</option>
                                </select>
                            </div>
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
                        </div>
                        
                        <div class="form-section">
                            <div class="section-title">👤 Sender Information</div>
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
                        
                        <div class="form-section">
                            <div class="section-title">👥 Receiver Information</div>
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
                        
                        <div class="form-section">
                            <div class="section-title">🏪 Transaction Context</div>
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
                        
                        <div class="form-section">
                            <div class="section-title">📱 Technical Details</div>
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
                    
                    <button type="submit" class="analyze-btn" id="analyzeBtn">
                        🔍 Analyze with Quantum AI Meta Model
                    </button>
                </form>
            </div>
            
            <div class="loading" id="loading">
                <div class="loading-spinner"></div>
                <p>Running comprehensive AI analysis...</p>
                <p><small>Quantum Computing + Machine Learning + Gemini AI</small></p>
            </div>
            
            <div class="results" id="results">
                <!-- Results will be populated by JavaScript -->
            </div>
        </div>
        
        <script>
            document.getElementById('fraudForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const loading = document.getElementById('loading');
                const results = document.getElementById('results');
                const analyzeBtn = document.getElementById('analyzeBtn');
                
                // Show loading
                loading.style.display = 'block';
                results.style.display = 'none';
                analyzeBtn.disabled = true;
                
                // Collect form data
                const formData = {
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
                        body: JSON.stringify(formData)
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const result = await response.json();
                    displayResults(result);
                    
                } catch (error) {
                    console.error('Error:', error);
                    alert('Error analyzing transaction: ' + error.message);
                } finally {
                    loading.style.display = 'none';
                    analyzeBtn.disabled = false;
                }
            });
            
            function displayResults(result) {
                const results = document.getElementById('results');
                
                // Determine risk class
                const riskClass = getRiskClass(result.risk_level);
                
                results.innerHTML = `
                    <div class="results-header">
                        <h2>🎯 Meta AI Analysis Complete</h2>
                        <p>Comprehensive fraud analysis using 5 advanced models</p>
                    </div>
                    
                    <div class="final-score ${riskClass}">
                        <div style="font-size: 1.2em; margin-bottom: 10px;">Final Fraud Score</div>
                        <div style="font-size: 3em; font-weight: bold;">${result.final_fraud_score.toFixed(1)}%</div>
                        <div style="font-size: 1.1em; margin-top: 10px;">Risk Level: ${result.risk_level}</div>
                        <div style="font-size: 0.9em; opacity: 0.8;">Confidence: ${result.confidence_score.toFixed(1)}%</div>
                    </div>
                    
                    <div class="scores-grid">
                        <div class="score-card">
                            <div class="score-label">🔮 Quantum Score</div>
                            <div class="score-value">${result.quantum_score.toFixed(1)}%</div>
                            <div class="score-desc">4-qubit quantum kernel</div>
                        </div>
                        
                        <div class="score-card">
                            <div class="score-label">🤖 Classical Score</div>
                            <div class="score-value">${result.classical_score.toFixed(1)}%</div>
                            <div class="score-desc">XGBoost ML model</div>
                        </div>
                        
                        <div class="score-card">
                            <div class="score-label">⚡ Neuro-QKAD</div>
                            <div class="score-value">${result.neuro_qkad_fusion_score.toFixed(1)}%</div>
                            <div class="score-desc">Quantum-classical fusion</div>
                        </div>
                        
                        <div class="score-card">
                            <div class="score-label">🧠 Gemini AI</div>
                            <div class="score-value">${result.gemini_logical_score.toFixed(1)}%</div>
                            <div class="score-desc">AI logical analysis</div>
                        </div>
                    </div>
                    
                    <div class="insights-grid">
                        <div class="insight-card">
                            <div class="insight-title">🚨 Risk Factors Detected</div>
                            <ul class="risk-factors">
                                ${result.primary_risk_factors.map(factor => `<li>${factor}</li>`).join('')}
                            </ul>
                        </div>
                        
                        <div class="insight-card">
                            <div class="insight-title">🔍 Fraud Analysis</div>
                            <p><strong>Fraud Type:</strong> ${result.fraud_type_detected}</p>
                            <p><strong>Model Agreement:</strong> ${result.model_agreement.toFixed(1)}%</p>
                            <p><strong>Uncertainty:</strong> ${result.uncertainty_measure.toFixed(1)}%</p>
                            <p><strong>Processing Time:</strong> ${result.processing_time_ms.toFixed(0)}ms</p>
                        </div>
                        
                        <div class="insight-card">
                            <div class="insight-title">📊 Model Performance</div>
                            <p><strong>Model Version:</strong> ${result.model_version}</p>
                            <p><strong>Analysis Time:</strong> ${result.timestamp}</p>
                            <p><strong>Components:</strong> 5 AI models</p>
                        </div>
                    </div>
                    
                    <div class="recommended-action">
                        <div class="insight-title">💡 Recommended Action</div>
                        <p>${result.recommended_action}</p>
                    </div>
                `;
                
                results.style.display = 'block';
                results.scrollIntoView({ behavior: 'smooth' });
            }
            
            function getRiskClass(riskLevel) {
                const riskClasses = {
                    'CRITICAL': 'score-critical',
                    'HIGH': 'score-high',
                    'MEDIUM': 'score-medium',
                    'LOW': 'score-low',
                    'MINIMAL': 'score-minimal'
                };
                return riskClasses[riskLevel] || 'score-medium';
            }
            
            // Set current time and day
            document.addEventListener('DOMContentLoaded', function() {
                const now = new Date();
                document.getElementById('hour_of_day').value = now.getHours();
                
                const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
                document.getElementById('day_of_week').value = dayNames[now.getDay()];
                
                const isWeekend = [0, 6].includes(now.getDay()) ? 1 : 0;
                document.getElementById('is_weekend').value = isWeekend;
            });
        </script>
    </body>
    </html>
    """

@app.post("/predict", response_model=MetaPredictionResponse)
async def predict_fraud(transaction: TransactionInput):
    """
    Comprehensive fraud prediction using meta model
    """
    if meta_model is None:
        raise HTTPException(status_code=500, detail="Meta model not loaded")
    
    start_time = datetime.now()
    
    try:
        # Convert to dictionary
        transaction_data = transaction.dict()
        
        # Get meta model prediction
        prediction = meta_model.predict(transaction_data)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Build response
        response = MetaPredictionResponse(
            quantum_score=prediction.quantum_score,
            classical_score=prediction.classical_score,
            neuro_qkad_fusion_score=prediction.neuro_qkad_fusion_score,
            gemini_logical_score=prediction.gemini_logical_score,
            final_fraud_score=prediction.final_fraud_score,
            confidence_score=prediction.confidence_score,
            risk_level=prediction.risk_level,
            primary_risk_factors=prediction.primary_risk_factors,
            fraud_type_detected=prediction.fraud_type_detected,
            recommended_action=prediction.recommended_action,
            model_agreement=prediction.model_agreement,
            uncertainty_measure=prediction.uncertainty_measure,
            timestamp=prediction.timestamp,
            model_version=prediction.model_version,
            processing_time_ms=processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in fraud prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "meta_model_loaded": meta_model is not None,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.get("/model_info")
async def model_info():
    """Get comprehensive model information"""
    if meta_model is None:
        raise HTTPException(status_code=500, detail="Meta model not loaded")
    
    stats = meta_model.get_model_statistics()
    
    return {
        "title": "Meta Fraud Detection System",
        "description": "Advanced AI system combining quantum computing, machine learning, and logical analysis",
        "version": "2.0.0",
        "components": {
            "quantum_svm": "4-qubit quantum kernel SVM",
            "classical_ml": "XGBoost gradient boosting",
            "rule_based": "Domain-specific fraud rules",
            "gemini_ai": "Google Gemini logical analysis",
            "meta_fusion": "Advanced meta-learning fusion"
        },
        "features": {
            "total_features": 14,
            "numeric_features": 3,
            "categorical_features": 11
        },
        "model_statistics": stats,
        "api_endpoints": [
            {"endpoint": "/", "method": "GET", "description": "Web interface"},
            {"endpoint": "/predict", "method": "POST", "description": "Fraud prediction"},
            {"endpoint": "/health", "method": "GET", "description": "Health check"},
            {"endpoint": "/model_info", "method": "GET", "description": "Model information"}
        ]
    }

@app.get("/fraud_trends")
async def get_fraud_trends():
    """Get current fraud trends from Gemini AI"""
    if meta_model is None or meta_model.gemini_model is None:
        raise HTTPException(status_code=500, detail="Gemini model not available")
    
    try:
        trends = meta_model.gemini_model.get_fraud_trends()
        return trends
    except Exception as e:
        logger.error(f"Error getting fraud trends: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting trends: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")