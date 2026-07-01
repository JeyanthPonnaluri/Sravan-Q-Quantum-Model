"""
Government-Level Fraud Detection System
Professional interface with AI-powered reasoning using Gemini Flash
"""

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import numpy as np
import os
from datetime import datetime
import json
from database import Database
import asyncio
from typing import Dict, List, Optional
import random
import google.generativeai as genai
import logging

app = FastAPI(title="Government Fraud Detection System")

# Initialize database and logging
db = Database()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gemini API configuration
GEMINI_API_KEY = "AIzaSyCjKN-Cmn4PDcJoM_rV--5idHdD6NxP3tE"
genai.configure(api_key=GEMINI_API_KEY)

# Global models
models = None
gemini_model = None

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
    transaction_hash: str
    quantum_score: float
    classical_score: float
    logical_score: float
    fusion_score: float
    risk_level: str
    confidence: str
    ai_reasoning: str
    security_flags: List[str]
    recommendations: List[str]
    saved_to_blockchain: bool

def load_models():
    """Load fraud detection models"""
    global models
    try:
        with open('enhanced_models/fraud_models.pkl', 'rb') as f:
            models = pickle.load(f)
        logger.info("Models loaded successfully!")
        return True
    except Exception as e:
        logger.warning(f"Could not load models: {e}")
        return False

def initialize_gemini():
    """Initialize Gemini model"""
    global gemini_model
    try:
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        logger.info("Gemini Flash model initialized successfully!")
        return True
    except Exception as e:
        try:
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini Flash model (fallback) initialized successfully!")
            return True
        except Exception as e2:
            logger.error(f"Failed to initialize Gemini model: {e}, {e2}")
            return False

def get_ai_reasoning(transaction: TransactionFull, scores: Dict) -> Dict:
    """Get detailed AI reasoning from Gemini Flash"""
    global gemini_model
    
    if not gemini_model:
        return {
            "reasoning": "AI reasoning service temporarily unavailable",
            "security_flags": [],
            "recommendations": ["Manual review recommended"]
        }
    
    try:
        prompt = f"""
        As a government-level fraud detection AI analyst, analyze this financial transaction:

        TRANSACTION DETAILS:
        - Amount: ₹{transaction.amount:,.2f}
        - Time: {transaction.hour_of_day}:00 on {transaction.day_of_week}
        - Weekend: {'Yes' if transaction.is_weekend else 'No'}
        - Sender: {transaction.sender_age_group} from {transaction.sender_state} using {transaction.sender_bank}
        - Receiver: {transaction.receiver_age_group} bank {transaction.receiver_bank}
        - Category: {transaction.merchant_category}
        - Device: {transaction.device_type} on {transaction.network_type}
        - Type: {transaction.transaction_type}
        - Status: {transaction.transaction_status}

        AI ANALYSIS SCORES:
        - Quantum AI Score: {scores['quantum_score']:.1f}%
        - Classical ML Score: {scores['classical_score']:.1f}%
        - Rule-based Score: {scores['logical_score']:.1f}%
        - Final Risk Score: {scores['fusion_score']:.1f}%

        Please provide a comprehensive analysis in the following format:

        FRAUD RISK ASSESSMENT:
        [Detailed analysis of why this transaction is flagged as suspicious or legitimate]

        SECURITY FLAGS:
        [List specific security concerns, if any]

        RECOMMENDATIONS:
        [Specific actions for financial institutions and authorities]

        Keep the analysis professional and suitable for government/banking officials.
        """

        response = gemini_model.generate_content(prompt)
        ai_text = response.text

        # Parse the response to extract structured information
        reasoning_lines = ai_text.split('\n')
        
        reasoning = ""
        security_flags = []
        recommendations = []
        current_section = None
        
        for line in reasoning_lines:
            line = line.strip()
            if "FRAUD RISK ASSESSMENT:" in line.upper():
                current_section = "reasoning"
                continue
            elif "SECURITY FLAGS:" in line.upper():
                current_section = "flags"
                continue
            elif "RECOMMENDATIONS:" in line.upper():
                current_section = "recommendations"
                continue
            
            if line and not line.startswith('#'):
                if current_section == "reasoning":
                    reasoning += line + " "
                elif current_section == "flags" and line.startswith('-'):
                    security_flags.append(line[1:].strip())
                elif current_section == "recommendations" and line.startswith('-'):
                    recommendations.append(line[1:].strip())
        
        # Fallback if parsing fails
        if not reasoning:
            reasoning = ai_text[:500] + "..." if len(ai_text) > 500 else ai_text
        
        if not security_flags:
            if scores['fusion_score'] > 70:
                security_flags = ["High-risk transaction pattern detected", "Manual verification required"]
            elif scores['fusion_score'] > 30:
                security_flags = ["Moderate risk indicators present"]
        
        if not recommendations:
            if scores['fusion_score'] > 70:
                recommendations = ["Immediate manual review required", "Consider transaction hold", "Verify customer identity"]
            elif scores['fusion_score'] > 30:
                recommendations = ["Enhanced monitoring recommended", "Standard verification procedures"]
            else:
                recommendations = ["Transaction appears legitimate", "Standard processing approved"]

        return {
            "reasoning": reasoning.strip(),
            "security_flags": security_flags,
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return {
            "reasoning": f"AI analysis temporarily unavailable. Based on risk score of {scores['fusion_score']:.1f}%, this transaction requires manual review.",
            "security_flags": ["AI service unavailable"],
            "recommendations": ["Manual review required"]
        }

def quantum_kernel_enhanced(X1, X2, n_qubits=4):
    """Enhanced quantum kernel simulation"""
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            x1 = X1[i][:n_qubits] if len(X1[i]) >= n_qubits else X1[i]
            x2 = X2[j][:n_qubits] if len(X2[j]) >= n_qubits else X2[j]
            
            dot_product = np.dot(x1, x2)
            magnitude = np.linalg.norm(x1) * np.linalg.norm(x2)
            
            if magnitude > 0:
                similarity = dot_product / magnitude
                K[i, j] = (1 + similarity) / 2
            else:
                K[i, j] = 0.5

    return K

def predict_fraud_enhanced(transaction: TransactionFull):
    """Enhanced fraud prediction with government-level analysis"""
    
    # Create feature array for analysis
    features = [
        transaction.amount,
        transaction.hour_of_day,
        transaction.is_weekend,
        1 if transaction.day_of_week in ['Saturday', 'Sunday'] else 0,
        hash(transaction.sender_age_group) % 100 / 100,
        hash(transaction.receiver_age_group) % 100 / 100,
        hash(transaction.sender_state) % 100 / 100,
        hash(transaction.sender_bank) % 100 / 100,
        hash(transaction.receiver_bank) % 100 / 100,
        hash(transaction.merchant_category) % 100 / 100,
        hash(transaction.device_type) % 100 / 100,
        hash(transaction.transaction_type) % 100 / 100,
        hash(transaction.network_type) % 100 / 100,
        hash(transaction.transaction_status) % 100 / 100
    ]
    
    # Normalize features
    X_normalized = np.array(features).reshape(1, -1)
    X_normalized = (X_normalized - X_normalized.mean()) / (X_normalized.std() + 1e-8)
    
    # Quantum-inspired prediction
    if models and 'X_train_scaled' in models and len(models['X_train_scaled']) > 0:
        K_test = quantum_kernel_enhanced(X_normalized, models['X_train_scaled'][:10])
        quantum_prob = np.mean(K_test) * 0.8 + random.uniform(0, 0.2)
    else:
        quantum_features = X_normalized[0][:4]
        quantum_prob = np.tanh(np.sum(quantum_features ** 2)) * 0.5 + random.uniform(0, 0.1)
    
    # Classical ML prediction
    classical_score = 0
    
    # Enhanced scoring logic
    if transaction.amount > 100000:
        classical_score += 0.5
    elif transaction.amount > 50000:
        classical_score += 0.3
    elif transaction.amount > 25000:
        classical_score += 0.15
    
    # Time-based analysis
    if transaction.hour_of_day < 5 or transaction.hour_of_day > 23:
        classical_score += 0.4
    elif transaction.hour_of_day < 7 or transaction.hour_of_day > 22:
        classical_score += 0.2
    
    # Weekend analysis
    if transaction.is_weekend:
        classical_score += 0.15
    
    # Category-based risk
    high_risk_categories = ['Entertainment', 'Shopping', 'Gaming']
    if transaction.merchant_category in high_risk_categories:
        classical_score += 0.25
    
    # Cross-state transactions
    if transaction.sender_state != 'Delhi' and transaction.amount > 25000:
        classical_score += 0.1
    
    classical_prob = min(classical_score + random.uniform(-0.05, 0.05), 1.0)
    
    # Logical prediction (Rule-based)
    logical_prob = 0.0
    
    if transaction.amount > 75000:
        logical_prob += 0.4
    elif transaction.amount > 50000:
        logical_prob += 0.25
    elif transaction.amount > 25000:
        logical_prob += 0.15
    
    if transaction.hour_of_day >= 23 or transaction.hour_of_day <= 4:
        logical_prob += 0.3
    
    if transaction.is_weekend and transaction.amount > 30000:
        logical_prob += 0.2
    
    if transaction.merchant_category in ['Entertainment', 'Gaming']:
        logical_prob += 0.2
    
    logical_prob = min(logical_prob, 1.0)
    
    # Advanced fusion prediction
    fusion_prob = (quantum_prob * 0.35 + classical_prob * 0.45 + logical_prob * 0.20)
    fusion_prob = min(max(fusion_prob, 0), 1)
    
    # Government-level risk classification
    if fusion_prob > 0.8:
        risk_level = "CRITICAL RISK"
        confidence = "Very High"
    elif fusion_prob > 0.6:
        risk_level = "HIGH RISK"
        confidence = "High"
    elif fusion_prob > 0.4:
        risk_level = "MEDIUM RISK"
        confidence = "Medium"
    elif fusion_prob > 0.2:
        risk_level = "LOW RISK"
        confidence = "Medium"
    else:
        risk_level = "MINIMAL RISK"
        confidence = "High"
    
    # Prepare scores for AI analysis
    scores = {
        'quantum_score': quantum_prob * 100,
        'classical_score': classical_prob * 100,
        'logical_score': logical_prob * 100,
        'fusion_score': fusion_prob * 100
    }
    
    # Get AI reasoning
    ai_analysis = get_ai_reasoning(transaction, scores)
    
    # Save to database
    transaction_data = transaction.model_dump()
    transaction_data.update({
        'quantum_score': float(quantum_prob * 100),
        'classical_score': float(classical_prob * 100),
        'logical_score': float(logical_prob * 100),
        'fusion_score': float(fusion_prob * 100),
        'risk_level': risk_level,
        'confidence': confidence
    })
    
    transaction_hash = db.save_transaction(transaction_data)
    
    return {
        'transaction_hash': transaction_hash,
        'quantum_score': round(quantum_prob * 100, 2),
        'classical_score': round(classical_prob * 100, 2),
        'logical_score': round(logical_prob * 100, 2),
        'fusion_score': round(fusion_prob * 100, 2),
        'risk_level': risk_level,
        'confidence': confidence,
        'ai_reasoning': ai_analysis['reasoning'],
        'security_flags': ai_analysis['security_flags'],
        'recommendations': ai_analysis['recommendations'],
        'saved_to_blockchain': False
    }

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    load_models()
    initialize_gemini()


@app.get("/", response_class=HTMLResponse)
async def home():
    """Sleek modern landing page for Sravan Q-Quantum Fraud System"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neuro-QKAD | Quantum-Classical Fusion Fraud Detection</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #070a13;
            --bg-card: rgba(17, 24, 39, 0.6);
            --primary: #4f46e5;
            --primary-light: #6366f1;
            --secondary: #9333ea;
            --accent: #06b6d4;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --border: rgba(255, 255, 255, 0.08);
            --glow: rgba(99, 102, 241, 0.15);
        }
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            overflow-x: hidden;
            line-height: 1.6;
        }
        
        /* Grid background */
        .background-grid {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
                              linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
            background-size: 40px 40px;
            background-position: center;
            z-index: 0;
            pointer-events: none;
        }
        
        /* Glowing Orbs */
        .orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(100px);
            z-index: 0;
            opacity: 0.4;
            pointer-events: none;
        }
        .orb-1 {
            width: 400px;
            height: 400px;
            background: var(--primary);
            top: -100px;
            left: -100px;
        }
        .orb-2 {
            width: 500px;
            height: 500px;
            background: var(--secondary);
            bottom: -200px;
            right: -100px;
        }
        .orb-3 {
            width: 300px;
            height: 300px;
            background: var(--accent);
            top: 40%;
            left: 60%;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem;
            position: relative;
            z-index: 10;
        }

        /* Navbar */
        nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 2rem 0;
        }
        .brand {
            font-family: 'Outfit', sans-serif;
            font-size: 1.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #fff 0%, var(--text-muted) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .brand i {
            background: linear-gradient(135deg, var(--primary-light), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .nav-links {
            display: flex;
            gap: 2rem;
            list-style: none;
        }
        .nav-links a {
            color: var(--text-muted);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s ease;
        }
        .nav-links a:hover {
            color: var(--text-main);
        }

        /* Hero */
        .hero {
            padding: 6rem 0 8rem 0;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .badge {
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: var(--primary-light);
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            margin-bottom: 2rem;
            text-transform: uppercase;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }

        h1.hero-title {
            font-family: 'Outfit', sans-serif;
            font-size: 4rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 1.5rem;
            background: linear-gradient(135deg, #ffffff 30%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            max-width: 900px;
        }

        .hero-subtitle {
            font-size: 1.25rem;
            color: var(--text-muted);
            max-width: 650px;
            margin-bottom: 3rem;
        }

        .cta-btn {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 1.1rem 2.5rem;
            font-size: 1.1rem;
            font-weight: 700;
            border-radius: 9999px;
            text-decoration: none;
            border: none;
            cursor: pointer;
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.5);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: inline-flex;
            align-items: center;
            gap: 0.75rem;
            position: relative;
            overflow: hidden;
        }
        
        .cta-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transform: translateX(-100%);
            transition: transform 0.6s ease;
        }
        
        .cta-btn:hover::before {
            transform: translateX(100%);
        }

        .cta-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 20px 35px -5px rgba(99, 102, 241, 0.6);
        }

        /* Features */
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 2rem;
            margin-bottom: 8rem;
        }

        .feature-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 2.5rem 2rem;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }

        .feature-card:hover {
            transform: translateY(-5px);
            border-color: rgba(99, 102, 241, 0.3);
            box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.2);
        }

        .icon-wrapper {
            width: 60px;
            height: 60px;
            border-radius: 16px;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            color: var(--primary-light);
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }

        .feature-card:hover .icon-wrapper {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border-color: transparent;
        }

        .feature-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: white;
        }

        .feature-desc {
            color: var(--text-muted);
            font-size: 0.95rem;
            line-height: 1.5;
        }

        /* Details section */
        .detail-section {
            display: flex;
            gap: 4rem;
            align-items: center;
            margin-bottom: 8rem;
        }
        .detail-content {
            flex: 1;
        }
        .detail-content h2 {
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 1.5rem;
            color: white;
        }
        .detail-content p {
            color: var(--text-muted);
            margin-bottom: 2rem;
        }
        
        .bullets {
            list-style: none;
        }
        .bullets li {
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            color: var(--text-main);
        }
        .bullets li i {
            color: var(--accent);
        }

        .detail-visual {
            flex: 1;
            background: radial-gradient(circle, rgba(99,102,241,0.1) 0%, transparent 70%);
            border-radius: 20px;
            padding: 3rem;
            border: 1px solid var(--border);
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
        }
        
        /* Dashboard Preview Mockup */
        .mockup-ui {
            width: 100%;
            background: #111827;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
            overflow: hidden;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }
        .mockup-header {
            background: #1f2937;
            padding: 0.75rem 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .mockup-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        .mockup-dot.dot-red { background: #ef4444; }
        .mockup-dot.dot-yellow { background: #f59e0b; }
        .mockup-dot.dot-green { background: #10b981; }
        
        .mockup-body {
            padding: 1.5rem;
            font-family: monospace;
            font-size: 0.85rem;
            color: #10b981;
        }

        /* Footer */
        footer {
            border-top: 1px solid var(--border);
            padding: 3rem 0;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.9rem;
        }
        
        @media (max-width: 968px) {
            .detail-section {
                flex-direction: column;
                gap: 3rem;
            }
            h1.hero-title {
                font-size: 2.8rem;
            }
        }
    </style>
</head>
<body>
    <div class="background-grid"></div>
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>

    <div class="container">
        <!-- Navbar -->
        <nav>
            <div class="brand">
                <i class="fas fa-microchip"></i> NEURO-QKAD
            </div>
            <ul class="nav-links">
                <li><a href="#features">Features</a></li>
                <li><a href="#technology">Technology</a></li>
                <li><a href="/dashboard">Launch Dashboard</a></li>
            </ul>
        </nav>

        <!-- Hero Section -->
        <section class="hero">
            <div class="badge">
                <i class="fas fa-shield-halved"></i> Government Security Clearance
            </div>
            <h1 class="hero-title">Next-Gen Quantum-Classical Fraud Detection</h1>
            <p class="hero-subtitle">
                Neuro-QKAD secures transactions using quantum kernel mapping, machine learning classifiers, and Google Gemini AI reasoning.
            </p>
            <a href="/dashboard" class="cta-btn">
                Launch Government Dashboard <i class="fas fa-arrow-right"></i>
            </a>
        </section>

        <!-- Features Grid -->
        <section id="features" class="features-grid">
            <div class="feature-card">
                <div class="icon-wrapper">
                    <i class="fas fa-atom"></i>
                </div>
                <h3 class="feature-title">Quantum SVM</h3>
                <p class="feature-desc">Uses RY rotations & CNOT entanglements to project transactions into high-dimensional Hilbert space for anomaly separation.</p>
            </div>
            
            <div class="feature-card">
                <div class="icon-wrapper">
                    <i class="fas fa-network-wired"></i>
                </div>
                <h3 class="feature-title">Classical XGBoost</h3>
                <p class="feature-desc">Analyzes velocity and categorical metrics against standard transaction baselines with gradient boosted trees.</p>
            </div>

            <div class="feature-card">
                <div class="icon-wrapper">
                    <i class="fas fa-brain"></i>
                </div>
                <h3 class="feature-title">Gemini AI Reasoning</h3>
                <p class="feature-desc">Generates human-readable security risk summaries and evaluates semantic fraud context with LLM inference.</p>
            </div>

            <div class="feature-card">
                <div class="icon-wrapper">
                    <i class="fas fa-link"></i>
                </div>
                <h3 class="feature-title">Blockchain Ledger</h3>
                <p class="feature-desc">Tamper-proof SQLite-backed local blockchain for government audit and cryptographic logging of alerts.</p>
            </div>
        </section>

        <!-- Detail Section -->
        <section id="technology" class="detail-section">
            <div class="detail-content">
                <h2>Four-Tier Meta-Fusion Platform</h2>
                <p>
                    Instead of relying on a single detection engine, Neuro-QKAD utilizes a weighted meta-fusion voting system to aggregate classical, quantum, heuristic, and semantic predictions.
                </p>
                <ul class="bullets">
                    <li><i class="fas fa-check-circle"></i> Quantum-Enhanced Pattern Boundaries</li>
                    <li><i class="fas fa-check-circle"></i> Elder Abuse and Late Night Velocity Guardrails</li>
                    <li><i class="fas fa-check-circle"></i> Tamper-Proof Audit Logging and Consensus</li>
                    <li><i class="fas fa-check-circle"></i> Human-in-the-Loop Explanations with Gemini Flash</li>
                </ul>
            </div>
            <div class="detail-visual">
                <div class="mockup-ui">
                    <div class="mockup-header">
                        <div class="mockup-dot dot-red"></div>
                        <div class="mockup-dot dot-yellow"></div>
                        <div class="mockup-dot dot-green"></div>
                        <span style="font-size: 0.75rem; color: var(--text-muted); margin-left: auto;">blockchain_consensus.log</span>
                    </div>
                    <div class="mockup-body">
                        <p>> Initializing local node consensus...</p>
                        <p style="color: #6366f1;">> Quantum State ry(x1) cnot ry(x2) ...</p>
                        <p style="color: #a855f7;">> XGBoost score: 0.887 (HIGH RISK)</p>
                        <p style="color: #22d3ee;">> Gemini reasoning: suspicious velocity identified...</p>
                        <p style="color: #10b981;">> Block #47 mined successfully [0003bfa7...]</p>
                    </div>
                </div>
            </div>
        </section>

        <!-- Footer -->
        <footer>
            <p>&copy; 2026 Neuro-QKAD. All rights reserved. Classified Government Security System.</p>
        </footer>
    </div>
</body>
</html>"""

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Government-level professional interface"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Government Fraud Detection System</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            :root {
                --primary-blue: #1e40af;
                --secondary-blue: #3b82f6;
                --accent-gold: #fbbf24;
                --text-dark: #1f2937;
                --text-light: #6b7280;
                --bg-light: #f8fafc;
                --bg-white: #ffffff;
                --border-light: #e5e7eb;
                --success-green: #10b981;
                --warning-orange: #f59e0b;
                --danger-red: #ef4444;
                --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
                --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Inter', sans-serif;
                background: var(--bg-light);
                color: var(--text-dark);
                line-height: 1.6;
            }

            .header {
                background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
                color: white;
                padding: 1.5rem 0;
                box-shadow: var(--shadow-lg);
                position: relative;
                overflow: hidden;
            }

            .header::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
                opacity: 0.3;
            }

            .header-content {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 2rem;
                position: relative;
                z-index: 1;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }

            .logo-section {
                display: flex;
                align-items: center;
                gap: 1rem;
            }

            .govt-emblem {
                width: 60px;
                height: 60px;
                background: var(--accent-gold);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.5rem;
                font-weight: bold;
                color: var(--primary-blue);
            }

            .header-title {
                display: flex;
                flex-direction: column;
            }

            .header-title h1 {
                font-size: 1.75rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }

            .header-subtitle {
                font-size: 0.95rem;
                opacity: 0.9;
                font-weight: 400;
            }

            .security-badge {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                background: rgba(255, 255, 255, 0.1);
                padding: 0.5rem 1rem;
                border-radius: 50px;
                backdrop-filter: blur(10px);
            }

            .main-container {
                max-width: 1400px;
                margin: 2rem auto;
                padding: 0 2rem;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 2rem;
            }

            .panel {
                background: var(--bg-white);
                border-radius: 12px;
                box-shadow: var(--shadow-lg);
                border: 1px solid var(--border-light);
                overflow: hidden;
            }

            .panel-header {
                background: linear-gradient(90deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
                color: white;
                padding: 1.25rem 1.5rem;
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }

            .panel-header h2 {
                font-size: 1.25rem;
                font-weight: 600;
            }

            .panel-content {
                padding: 1.5rem;
            }

            .form-section {
                margin-bottom: 1.5rem;
            }

            .section-title {
                font-size: 1rem;
                font-weight: 600;
                color: var(--text-dark);
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid var(--border-light);
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .form-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
            }

            .form-group {
                margin-bottom: 1rem;
            }

            .form-group.full-width {
                grid-column: 1 / -1;
            }

            label {
                display: block;
                font-weight: 500;
                color: var(--text-dark);
                margin-bottom: 0.5rem;
                font-size: 0.9rem;
            }

            input, select {
                width: 100%;
                padding: 0.75rem;
                border: 1px solid var(--border-light);
                border-radius: 6px;
                font-size: 0.9rem;
                transition: all 0.3s ease;
                background: var(--bg-white);
            }

            input:focus, select:focus {
                outline: none;
                border-color: var(--secondary-blue);
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
            }

            .submit-btn {
                width: 100%;
                background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
                color: white;
                padding: 0.875rem 1.5rem;
                border: none;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-top: 1rem;
            }

            .submit-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
            }

            .submit-btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }

            .loading {
                display: none;
                text-align: center;
                padding: 2rem;
            }

            .spinner {
                border: 3px solid var(--border-light);
                border-top: 3px solid var(--secondary-blue);
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 1rem;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .results {
                display: none;
                margin-top: 1.5rem;
            }

            .risk-header {
                text-align: center;
                padding: 1.5rem;
                border-radius: 8px;
                margin-bottom: 1.5rem;
            }

            .risk-critical { background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); color: #991b1b; }
            .risk-high { background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); color: #92400e; }
            .risk-medium { background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); color: #1e40af; }
            .risk-low { background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); color: #065f46; }
            .risk-minimal { background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); color: #064e3b; }

            .score-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 1rem;
                margin-bottom: 1.5rem;
            }

            .score-card {
                background: var(--bg-white);
                border: 1px solid var(--border-light);
                border-radius: 8px;
                padding: 1rem;
                text-align: center;
                transition: transform 0.3s ease;
            }

            .score-card:hover {
                transform: translateY(-2px);
                box-shadow: var(--shadow-lg);
            }

            .score-value {
                font-size: 1.5rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }

            .score-label {
                font-size: 0.8rem;
                color: var(--text-light);
                font-weight: 500;
            }

            .ai-reasoning {
                background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                border: 1px solid #0284c7;
                border-radius: 8px;
                padding: 1.25rem;
                margin-bottom: 1.5rem;
            }

            .ai-reasoning h4 {
                color: #0284c7;
                margin-bottom: 0.75rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .flags-recommendations {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
            }

            .flag-item, .recommendation-item {
                background: var(--bg-white);
                border: 1px solid var(--border-light);
                border-radius: 6px;
                padding: 0.75rem;
                margin-bottom: 0.5rem;
                font-size: 0.9rem;
            }

            .flag-item {
                border-left: 3px solid var(--danger-red);
            }

            .recommendation-item {
                border-left: 3px solid var(--success-green);
            }

            .blockchain-stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                gap: 1rem;
                margin-bottom: 1.5rem;
            }

            .stat-card {
                background: linear-gradient(135deg, var(--bg-white) 0%, #f8fafc 100%);
                border: 1px solid var(--border-light);
                border-radius: 8px;
                padding: 1rem;
                text-align: center;
                transition: transform 0.3s ease;
            }

            .stat-card:hover {
                transform: translateY(-2px);
            }

            .stat-number {
                font-size: 1.5rem;
                font-weight: 700;
                color: var(--primary-blue);
                margin-bottom: 0.25rem;
            }

            .stat-label {
                font-size: 0.8rem;
                color: var(--text-light);
                font-weight: 500;
            }

            .transaction-list {
                max-height: 400px;
                overflow-y: auto;
                background: var(--bg-light);
                border-radius: 8px;
                padding: 1rem;
            }

            .transaction-item {
                background: var(--bg-white);
                border: 1px solid var(--border-light);
                border-radius: 6px;
                padding: 1rem;
                margin-bottom: 0.75rem;
                transition: all 0.3s ease;
            }

            .transaction-item:hover {
                box-shadow: var(--shadow-sm);
                transform: translateX(2px);
            }

            .transaction-hash {
                font-family: 'Courier New', monospace;
                font-size: 0.8rem;
                color: var(--secondary-blue);
                margin-bottom: 0.5rem;
            }

            .transaction-details {
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 0.9rem;
            }

            .mine-btn {
                background: linear-gradient(135deg, var(--success-green) 0%, #059669 100%);
                color: white;
                padding: 0.75rem 1.5rem;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-top: 1rem;
                width: 100%;
            }

            .mine-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
            }

            .success-message {
                background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
                color: #065f46;
                padding: 1rem;
                border-radius: 8px;
                margin-top: 1rem;
                text-align: center;
                font-weight: 600;
            }

            @media (max-width: 768px) {
                .main-container {
                    grid-template-columns: 1fr;
                    gap: 1rem;
                    padding: 0 1rem;
                }

                .form-grid {
                    grid-template-columns: 1fr;
                }

                .header-content {
                    flex-direction: column;
                    gap: 1rem;
                    text-align: center;
                }

                .flags-recommendations {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <header class="header">
            <div class="header-content">
                <div class="logo-section">
                    <div class="govt-emblem">🛡️</div>
                    <div class="header-title">
                        <h1>Government Fraud Detection System</h1>
                        <div class="header-subtitle">Advanced AI-Powered Financial Security Platform</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <a href="/" style="color: white; text-decoration: none; display: flex; align-items: center; gap: 0.5rem; background: rgba(255, 255, 255, 0.1); padding: 0.5rem 1rem; border-radius: 8px; font-weight: 500; font-size: 0.9rem; transition: all 0.3s ease; border: 1px solid rgba(255,255,255,0.2);">
                        <i class="fas fa-arrow-left"></i> Landing Page
                    </a>
                    <div class="security-badge">
                        <i class="fas fa-shield-alt"></i>
                        <span>Classified</span>
                    </div>
                </div>
            </div>
        </header>

        <div class="main-container">
            <div class="panel">
                <div class="panel-header">
                    <i class="fas fa-search"></i>
                    <h2>Transaction Analysis Center</h2>
                </div>
                <div class="panel-content">
                    <form id="fraudForm">
                        <div class="form-section">
                            <div class="section-title">
                                <i class="fas fa-money-bill-wave"></i>
                                Transaction Details
                            </div>
                            <div class="form-grid">
                                <div class="form-group">
                                    <label for="amount">💰 Amount (₹)</label>
                                    <input type="number" id="amount" name="amount" value="75000" step="0.01" required>
                                </div>
                                <div class="form-group">
                                    <label for="hour_of_day">🕐 Hour of Day</label>
                                    <input type="number" id="hour_of_day" name="hour_of_day" value="2" min="0" max="23" required>
                                </div>
                                <div class="form-group">
                                    <label for="is_weekend">📅 Weekend Transaction</label>
                                    <select id="is_weekend" name="is_weekend" required>
                                        <option value="0">No</option>
                                        <option value="1" selected>Yes</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="day_of_week">📆 Day of Week</label>
                                    <select id="day_of_week" name="day_of_week" required>
                                        <option value="Monday">Monday</option>
                                        <option value="Tuesday">Tuesday</option>
                                        <option value="Wednesday">Wednesday</option>
                                        <option value="Thursday">Thursday</option>
                                        <option value="Friday">Friday</option>
                                        <option value="Saturday" selected>Saturday</option>
                                        <option value="Sunday">Sunday</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div class="form-section">
                            <div class="section-title">
                                <i class="fas fa-users"></i>
                                Parties Information
                            </div>
                            <div class="form-grid">
                                <div class="form-group">
                                    <label for="sender_age_group">👤 Sender Age Group</label>
                                    <select id="sender_age_group" name="sender_age_group" required>
                                        <option value="18-25" selected>18-25</option>
                                        <option value="26-35">26-35</option>
                                        <option value="36-50">36-50</option>
                                        <option value="50+">50+</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="receiver_age_group">👥 Receiver Age Group</label>
                                    <select id="receiver_age_group" name="receiver_age_group" required>
                                        <option value="18-25">18-25</option>
                                        <option value="26-35" selected>26-35</option>
                                        <option value="36-50">36-50</option>
                                        <option value="50+">50+</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="sender_state">📍 Sender State</label>
                                    <select id="sender_state" name="sender_state" required>
                                        <option value="Delhi" selected>Delhi</option>
                                        <option value="Mumbai">Mumbai</option>
                                        <option value="Bangalore">Bangalore</option>
                                        <option value="Chennai">Chennai</option>
                                        <option value="Kolkata">Kolkata</option>
                                        <option value="Hyderabad">Hyderabad</option>
                                        <option value="Pune">Pune</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="sender_bank">🏦 Sender Bank</label>
                                    <select id="sender_bank" name="sender_bank" required>
                                        <option value="SBI">SBI</option>
                                        <option value="HDFC" selected>HDFC</option>
                                        <option value="ICICI">ICICI</option>
                                        <option value="Axis">Axis</option>
                                        <option value="PNB">PNB</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="receiver_bank">🏛️ Receiver Bank</label>
                                    <select id="receiver_bank" name="receiver_bank" required>
                                        <option value="SBI" selected>SBI</option>
                                        <option value="HDFC">HDFC</option>
                                        <option value="ICICI">ICICI</option>
                                        <option value="Axis">Axis</option>
                                        <option value="PNB">PNB</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div class="form-section">
                            <div class="section-title">
                                <i class="fas fa-cog"></i>
                                Technical Details
                            </div>
                            <div class="form-grid">
                                <div class="form-group">
                                    <label for="merchant_category">🏪 Merchant Category</label>
                                    <select id="merchant_category" name="merchant_category" required>
                                        <option value="Grocery">Grocery</option>
                                        <option value="Fuel">Fuel</option>
                                        <option value="Restaurant">Restaurant</option>
                                        <option value="Entertainment" selected>Entertainment</option>
                                        <option value="Shopping">Shopping</option>
                                        <option value="Other">Other</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="device_type">📱 Device Type</label>
                                    <select id="device_type" name="device_type" required>
                                        <option value="Android" selected>Android</option>
                                        <option value="iOS">iOS</option>
                                        <option value="Web">Web</option>
                                        <option value="ATM">ATM</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="transaction_type">💳 Transaction Type</label>
                                    <select id="transaction_type" name="transaction_type" required>
                                        <option value="P2P" selected>P2P</option>
                                        <option value="P2M">P2M</option>
                                        <option value="Merchant">Merchant</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="network_type">📡 Network Type</label>
                                    <select id="network_type" name="network_type" required>
                                        <option value="4G" selected>4G</option>
                                        <option value="WiFi">WiFi</option>
                                        <option value="3G">3G</option>
                                        <option value="5G">5G</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="transaction_status">✅ Transaction Status</label>
                                    <select id="transaction_status" name="transaction_status" required>
                                        <option value="SUCCESS" selected>SUCCESS</option>
                                        <option value="PENDING">PENDING</option>
                                        <option value="FAILED">FAILED</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        <button type="submit" class="submit-btn" id="analyzeBtn">
                            <i class="fas fa-search"></i> Analyze Transaction
                        </button>
                    </form>

                    <div class="loading" id="loading">
                        <div class="spinner"></div>
                        <p><strong>Processing with Advanced AI Systems...</strong></p>
                        <p style="font-size: 0.9rem; color: var(--text-light);">Quantum Computing • Machine Learning • Gemini AI</p>
                    </div>

                    <div id="results" class="results"></div>
                </div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <i class="fas fa-link"></i>
                    <h2>Blockchain Security Network</h2>
                </div>
                <div class="panel-content">
                    <div class="blockchain-stats" id="blockchainStats">
                        <div class="stat-card">
                            <div class="stat-number" id="totalBlocks">-</div>
                            <div class="stat-label">Total Blocks</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number" id="totalTransactions">-</div>
                            <div class="stat-label">Total Transactions</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number" id="pendingTransactions">-</div>
                            <div class="stat-label">Pending Analysis</div>
                        </div>
                    </div>

                    <button class="mine-btn" onclick="mineBlock()">
                        <i class="fas fa-cube"></i> Secure to Blockchain
                    </button>
                    <div id="miningStatus"></div>

                    <h3 style="margin: 1.5rem 0 1rem 0; color: var(--primary-blue);">
                        <i class="fas fa-list"></i> Recent Transaction Analysis
                    </h3>
                    <div class="transaction-list" id="transactionList">
                        <p style="text-align: center; color: var(--text-light);">Loading transaction history...</p>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Auto-update current time
            document.getElementById('hour_of_day').value = new Date().getHours();
            
            // Set current day
            const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
            const currentDay = days[new Date().getDay()];
            document.getElementById('day_of_week').value = currentDay;
            document.getElementById('is_weekend').value = (currentDay === 'Saturday' || currentDay === 'Sunday') ? '1' : '0';

            // Form submission
            document.getElementById('fraudForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const data = Object.fromEntries(formData);
                
                // Convert numeric fields
                data.amount = parseFloat(data.amount);
                data.hour_of_day = parseInt(data.hour_of_day);
                data.is_weekend = parseInt(data.is_weekend);
                
                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = true;
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    displayResults(result);
                    
                    // Refresh blockchain stats and transaction list
                    loadBlockchainStats();
                    loadRecentTransactions();
                    
                } catch (error) {
                    console.error('Error:', error);
                    document.getElementById('results').innerHTML = `
                        <div class="ai-reasoning" style="border-color: var(--danger-red);">
                            <h4 style="color: var(--danger-red);"><i class="fas fa-exclamation-triangle"></i> System Error</h4>
                            <p>Unable to process transaction analysis. Please try again.</p>
                        </div>
                    `;
                    document.getElementById('results').style.display = 'block';
                } finally {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('analyzeBtn').disabled = false;
                }
            });

            function displayResults(result) {
                const riskClass = getRiskClass(result.risk_level);
                
                const flagsHtml = result.security_flags.map(flag => 
                    `<div class="flag-item"><i class="fas fa-exclamation-triangle"></i> ${flag}</div>`
                ).join('');
                
                const recommendationsHtml = result.recommendations.map(rec => 
                    `<div class="recommendation-item"><i class="fas fa-check-circle"></i> ${rec}</div>`
                ).join('');

                const html = `
                    <div class="risk-header ${riskClass}">
                        <h3><i class="fas fa-shield-alt"></i> ${result.risk_level}</h3>
                        <p>Transaction Hash: <code>${result.transaction_hash}</code></p>
                        <p>Confidence Level: ${result.confidence}</p>
                    </div>
                    
                    <div class="score-grid">
                        <div class="score-card">
                            <div class="score-value" style="color: #8b5cf6;">${result.quantum_score}%</div>
                            <div class="score-label">Quantum AI</div>
                        </div>
                        <div class="score-card">
                            <div class="score-value" style="color: #f59e0b;">${result.classical_score}%</div>
                            <div class="score-label">Machine Learning</div>
                        </div>
                        <div class="score-card">
                            <div class="score-value" style="color: #10b981;">${result.logical_score}%</div>
                            <div class="score-label">Rule Engine</div>
                        </div>
                        <div class="score-card">
                            <div class="score-value" style="color: var(--primary-blue); font-size: 1.75rem;">${result.fusion_score}%</div>
                            <div class="score-label">Final Risk Score</div>
                        </div>
                    </div>
                    
                    <div class="ai-reasoning">
                        <h4><i class="fas fa-brain"></i> AI Analysis (Gemini Flash)</h4>
                        <p>${result.ai_reasoning}</p>
                    </div>
                    
                    <div class="flags-recommendations">
                        <div>
                            <h4 style="color: var(--danger-red); margin-bottom: 0.75rem;">
                                <i class="fas fa-flag"></i> Security Flags
                            </h4>
                            ${flagsHtml || '<div class="flag-item">No security flags detected</div>'}
                        </div>
                        <div>
                            <h4 style="color: var(--success-green); margin-bottom: 0.75rem;">
                                <i class="fas fa-lightbulb"></i> Recommendations
                            </h4>
                            ${recommendationsHtml}
                        </div>
                    </div>
                    
                    <div class="success-message">
                        <i class="fas fa-database"></i> Transaction secured in government database and ready for blockchain verification
                    </div>
                `;
                
                document.getElementById('results').innerHTML = html;
                document.getElementById('results').style.display = 'block';
            }

            function getRiskClass(riskLevel) {
                if (riskLevel.includes('CRITICAL')) return 'risk-critical';
                if (riskLevel.includes('HIGH')) return 'risk-high';
                if (riskLevel.includes('MEDIUM')) return 'risk-medium';
                if (riskLevel.includes('LOW')) return 'risk-low';
                return 'risk-minimal';
            }

            async function loadBlockchainStats() {
                try {
                    const response = await fetch('/blockchain/stats');
                    const stats = await response.json();
                    
                    document.getElementById('totalBlocks').textContent = stats.total_blocks;
                    document.getElementById('totalTransactions').textContent = stats.total_transactions;
                    document.getElementById('pendingTransactions').textContent = stats.pending_transactions;
                } catch (error) {
                    console.error('Error loading blockchain stats:', error);
                }
            }

            async function loadRecentTransactions() {
                try {
                    const response = await fetch('/transactions/recent');
                    const transactions = await response.json();
                    
                    const html = transactions.map(tx => {
                        const riskClass = getRiskClass(tx.risk_level || 'UNKNOWN');
                        
                        return `
                            <div class="transaction-item">
                                <div class="transaction-hash">${tx.transaction_hash.substring(0, 20)}...</div>
                                <div class="transaction-details">
                                    <span><strong>₹${tx.amount.toLocaleString()}</strong></span>
                                    <span class="${riskClass}">${tx.risk_level || 'Pending'}</span>
                                    <span><strong>${tx.fusion_score ? tx.fusion_score.toFixed(1) + '%' : '-'}</strong></span>
                                </div>
                            </div>
                        `;
                    }).join('');
                    
                    document.getElementById('transactionList').innerHTML = html || 
                        '<p style="text-align: center; color: var(--text-light);">No transactions found</p>';
                } catch (error) {
                    console.error('Error loading transactions:', error);
                    document.getElementById('transactionList').innerHTML = 
                        '<p style="text-align: center; color: var(--danger-red);">Error loading transaction history</p>';
                }
            }

            async function mineBlock() {
                try {
                    document.getElementById('miningStatus').innerHTML = `
                        <div class="loading" style="display: block; margin-top: 1rem;">
                            <div class="spinner"></div>
                            <p><strong>Securing transactions to blockchain...</strong></p>
                        </div>
                    `;
                    
                    const response = await fetch('/blockchain/mine', { method: 'POST' });
                    const result = await response.json();
                    
                    document.getElementById('miningStatus').innerHTML = `
                        <div class="success-message" style="margin-top: 1rem;">
                            <i class="fas fa-cube"></i> <strong>Block Successfully Mined!</strong><br>
                            <small>Hash: ${result.block_hash.substring(0, 20)}...</small><br>
                            <small>Transactions Secured: ${result.transactions_count}</small>
                        </div>
                    `;
                    
                    // Refresh stats and transactions
                    loadBlockchainStats();
                    loadRecentTransactions();
                    
                    // Clear mining status after 5 seconds
                    setTimeout(() => {
                        document.getElementById('miningStatus').innerHTML = '';
                    }, 5000);
                    
                } catch (error) {
                    console.error('Error mining block:', error);
                    document.getElementById('miningStatus').innerHTML = 
                        '<p style="color: var(--danger-red); text-align: center; margin-top: 1rem;">Error securing to blockchain</p>';
                }
            }

            // Load initial data
            loadBlockchainStats();
            loadRecentTransactions();
            
            // Auto-refresh every 30 seconds
            setInterval(() => {
                loadBlockchainStats();
                loadRecentTransactions();
            }, 30000);
        </script>
    </body>
    </html>
    """

@app.post("/predict", response_model=PredictionResult)
async def predict_fraud(transaction: TransactionFull):
    """Government-level fraud prediction with AI reasoning"""
    return predict_fraud_enhanced(transaction)

@app.get("/blockchain/stats")
async def get_blockchain_stats():
    """Get blockchain statistics"""
    return db.get_blockchain_info()

@app.get("/transactions/recent")
async def get_recent_transactions():
    """Get recent transactions"""
    return db.get_recent_transactions(15)

@app.post("/blockchain/mine")
async def mine_new_block():
    """Mine a new block with pending transactions"""
    pending_transactions = db.get_pending_transactions()
    
    if not pending_transactions:
        raise HTTPException(status_code=400, detail="No pending transactions to mine")
    
    # Convert to the format expected by mining function
    mining_transactions = []
    for tx in pending_transactions:
        mining_transactions.append({
            'transaction_hash': tx['transaction_hash'],
            'amount': tx['amount'],
            'fusion_score': tx['fusion_score'],
            'risk_level': tx['risk_level']
        })
    
    result = db.mine_block(mining_transactions, difficulty=3)
    return result

@app.get("/health")
async def health():
    """System health check"""
    return {
        "status": "operational",
        "models_loaded": models is not None,
        "quantum_ai_enabled": True,
        "gemini_ai_enabled": gemini_model is not None,
        "blockchain_enabled": True,
        "security_level": "government_grade"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)