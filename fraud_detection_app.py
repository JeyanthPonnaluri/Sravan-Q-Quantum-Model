"""
Advanced Fraud Detection System with Database and Blockchain Integration
"""

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import numpy as np
import pennylane as qml
import os
from datetime import datetime
import json
from database import Database
import asyncio
from typing import Dict, List

app = FastAPI(title="Advanced Fraud Detection with Blockchain")

# Initialize database
db = Database()

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
    transaction_hash: str
    quantum_score: float
    classical_score: float
    logical_score: float
    fusion_score: float
    risk_level: str
    confidence: str
    saved_to_blockchain: bool

def load_models():
    """Load the fraud detection models"""
    global models
    try:
        with open('enhanced_models/fraud_models.pkl', 'rb') as f:
            models = pickle.load(f)
        print("Models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

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
    
    feature_indices = {name: i for i, name in enumerate(feature_names)}
    
    for i, row in enumerate(X):
        score = 0.0
        
        # High amount transactions (>50000)
        if 'amount' in feature_indices:
            amount = row[feature_indices['amount']]
            if amount > 50000:
                score += 0.3
            elif amount > 25000:
                score += 0.15
        
        # Late night transactions (11 PM - 5 AM)
        if 'hour_of_day' in feature_indices:
            hour = row[feature_indices['hour_of_day']]
            if hour >= 23 or hour <= 5:
                score += 0.2
        
        # Weekend transactions
        if 'is_weekend' in feature_indices and row[feature_indices['is_weekend']] == 1:
            score += 0.1
        
        # High-risk categories
        if 'merchant_category_Entertainment' in feature_indices and row[feature_indices['merchant_category_Entertainment']] == 1:
            score += 0.15
        
        scores[i] = min(score, 1.0)
    
    return scores

def predict_fraud_enhanced(transaction: TransactionFull):
    """Predict fraud using all models"""
    if models is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    # Prepare input data
    data_dict = {
        'amount': [transaction.amount],
        'hour_of_day': [transaction.hour_of_day],
        'is_weekend': [transaction.is_weekend],
        'day_of_week': [transaction.day_of_week],
        'sender_age_group': [transaction.sender_age_group],
        'receiver_age_group': [transaction.receiver_age_group],
        'sender_state': [transaction.sender_state],
        'sender_bank': [transaction.sender_bank],
        'receiver_bank': [transaction.receiver_bank],
        'merchant_category': [transaction.merchant_category],
        'device_type': [transaction.device_type],
        'transaction_type': [transaction.transaction_type],
        'network_type': [transaction.network_type],
        'transaction_status': [transaction.transaction_status]
    }
    
    # Transform data using preprocessor
    X_transformed = models['preprocessor'].transform([list(data_dict.values())[i][0] for i in range(len(data_dict))])
    X_transformed = X_transformed.reshape(1, -1)
    
    # Quantum prediction
    if len(models['X_train_scaled']) > 0:
        K_test = quantum_kernel_enhanced(X_transformed, models['X_train_scaled'])
        quantum_prob = models['quantum_model'].predict_proba(K_test)[0, 1]
    else:
        quantum_prob = 0.0
    
    # Classical prediction
    classical_prob = models['classical_model'].predict_proba(X_transformed)[0, 1]
    
    # Logical prediction
    logical_scores = compute_logical_scores(X_transformed, models['feature_names'])
    logical_prob = logical_scores[0]
    
    # Fusion prediction
    fusion_features = np.array([[quantum_prob, classical_prob, logical_prob]])
    fusion_prob = models['fusion_model'].predict_proba(fusion_features)[0, 1]
    
    # Determine risk level
    if fusion_prob > 0.7:
        risk_level = "HIGH RISK"
        confidence = "High"
    elif fusion_prob > 0.3:
        risk_level = "MEDIUM RISK"
        confidence = "Medium"
    else:
        risk_level = "LOW RISK"
        confidence = "High"
    
    # Save to database
    transaction_data = transaction.dict()
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
        'saved_to_blockchain': False  # Will be True after mining
    }

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()

@app.get("/", response_class=HTMLResponse)
async def home():
    """Advanced web interface with blockchain visualization"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Advanced Fraud Detection with Blockchain</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                min-height: 100vh;
            }
            
            .header {
                text-align: center;
                padding: 20px;
                background: rgba(0,0,0,0.2);
                backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }
            
            .header h1 {
                font-size: 2.8em;
                margin-bottom: 10px;
                background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            
            .container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                padding: 20px;
                max-width: 1600px;
                margin: 0 auto;
            }
            
            .panel {
                background: rgba(255,255,255,0.1);
                padding: 25px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                border: 1px solid rgba(255,255,255,0.1);
            }
            
            .panel h2 {
                margin-bottom: 20px;
                color: #4ecdc4;
                font-size: 1.5em;
                border-bottom: 2px solid #4ecdc4;
                padding-bottom: 10px;
            }
            
            .form-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin-bottom: 20px;
            }
            
            .form-group {
                margin-bottom: 15px;
            }
            
            .form-group.full-width {
                grid-column: 1 / -1;
            }
            
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: 600;
                color: #e1e1e1;
            }
            
            input, select {
                width: 100%;
                padding: 12px;
                border: none;
                border-radius: 8px;
                background: rgba(255,255,255,0.9);
                color: #333;
                font-size: 14px;
                transition: all 0.3s ease;
            }
            
            input:focus, select:focus {
                outline: none;
                background: rgba(255,255,255,1);
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(78,205,196,0.3);
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
                transition: all 0.3s ease;
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(255,107,107,0.4);
            }
            
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .results {
                margin-top: 20px;
                padding: 20px;
                background: rgba(0,0,0,0.3);
                border-radius: 10px;
                border-left: 4px solid #4ecdc4;
            }
            
            .score-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 15px;
                margin: 15px 0;
            }
            
            .score-card {
                text-align: center;
                padding: 15px;
                background: rgba(255,255,255,0.1);
                border-radius: 8px;
                transition: transform 0.3s ease;
            }
            
            .score-card:hover {
                transform: scale(1.05);
            }
            
            .score-value {
                font-size: 1.8em;
                font-weight: bold;
                margin-bottom: 5px;
            }
            
            .score-label {
                font-size: 0.9em;
                opacity: 0.8;
            }
            
            .blockchain-stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            
            .stat-card {
                text-align: center;
                padding: 15px;
                background: linear-gradient(45deg, rgba(78,205,196,0.2), rgba(30,60,114,0.2));
                border-radius: 10px;
                border: 1px solid rgba(78,205,196,0.3);
            }
            
            .stat-number {
                font-size: 1.6em;
                font-weight: bold;
                color: #4ecdc4;
                margin-bottom: 5px;
            }
            
            .stat-label {
                font-size: 0.9em;
                opacity: 0.9;
            }
            
            .transaction-list {
                max-height: 400px;
                overflow-y: auto;
                background: rgba(0,0,0,0.2);
                border-radius: 8px;
                padding: 10px;
            }
            
            .transaction-item {
                background: rgba(255,255,255,0.1);
                margin-bottom: 10px;
                padding: 15px;
                border-radius: 8px;
                border-left: 3px solid #4ecdc4;
                transition: all 0.3s ease;
            }
            
            .transaction-item:hover {
                background: rgba(255,255,255,0.15);
                transform: translateX(5px);
            }
            
            .transaction-hash {
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                color: #4ecdc4;
                margin-bottom: 5px;
            }
            
            .transaction-details {
                display: flex;
                justify-content: space-between;
                font-size: 0.9em;
            }
            
            .risk-high { border-left-color: #ff6b6b; color: #ff6b6b; }
            .risk-medium { border-left-color: #ffa726; color: #ffa726; }
            .risk-low { border-left-color: #66bb6a; color: #66bb6a; }
            
            .mining-controls {
                margin-top: 20px;
                text-align: center;
            }
            
            .mine-button {
                background: linear-gradient(45deg, #4ecdc4, #44a08d);
                margin-top: 10px;
            }
            
            .mine-button:hover {
                box-shadow: 0 6px 20px rgba(78,205,196,0.4);
            }
            
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            
            .spinner {
                border: 3px solid rgba(255,255,255,0.3);
                border-radius: 50%;
                border-top: 3px solid #4ecdc4;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            @media (max-width: 768px) {
                .container {
                    grid-template-columns: 1fr;
                }
                
                .form-grid {
                    grid-template-columns: 1fr;
                }
                
                .header h1 {
                    font-size: 2.2em;
                }
            }
            
            .success-message {
                background: linear-gradient(45deg, #66bb6a, #4caf50);
                padding: 15px;
                border-radius: 8px;
                margin-top: 15px;
                text-align: center;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔐 Advanced Fraud Detection System</h1>
            <p>Quantum-Classical AI with Blockchain Integration</p>
        </div>
        
        <div class="container">
            <div class="panel">
                <h2>🚀 New Transaction Analysis</h2>
                <form id="fraudForm">
                    <div class="form-grid">
                        <div class="form-group">
                            <label for="amount">💰 Amount (₹)</label>
                            <input type="number" id="amount" name="amount" value="25000" step="0.01" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="hour_of_day">🕐 Hour of Day</label>
                            <input type="number" id="hour_of_day" name="hour_of_day" value="14" min="0" max="23" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="is_weekend">📅 Weekend?</label>
                            <select id="is_weekend" name="is_weekend" required>
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="day_of_week">📆 Day of Week</label>
                            <select id="day_of_week" name="day_of_week" required>
                                <option value="Monday">Monday</option>
                                <option value="Tuesday">Tuesday</option>
                                <option value="Wednesday">Wednesday</option>
                                <option value="Thursday" selected>Thursday</option>
                                <option value="Friday">Friday</option>
                                <option value="Saturday">Saturday</option>
                                <option value="Sunday">Sunday</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="sender_age_group">👤 Sender Age</label>
                            <select id="sender_age_group" name="sender_age_group" required>
                                <option value="18-25">18-25</option>
                                <option value="26-35" selected>26-35</option>
                                <option value="36-50">36-50</option>
                                <option value="50+">50+</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="receiver_age_group">👥 Receiver Age</label>
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
                                <option value="SBI" selected>SBI</option>
                                <option value="HDFC">HDFC</option>
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
                        
                        <div class="form-group">
                            <label for="merchant_category">🏪 Category</label>
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
                            <label for="device_type">📱 Device</label>
                            <select id="device_type" name="device_type" required>
                                <option value="Android" selected>Android</option>
                                <option value="iOS">iOS</option>
                                <option value="Web">Web</option>
                                <option value="ATM">ATM</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="transaction_type">💳 Type</label>
                            <select id="transaction_type" name="transaction_type" required>
                                <option value="P2P" selected>P2P</option>
                                <option value="P2M">P2M</option>
                                <option value="Merchant">Merchant</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="network_type">📡 Network</label>
                            <select id="network_type" name="network_type" required>
                                <option value="4G" selected>4G</option>
                                <option value="WiFi">WiFi</option>
                                <option value="3G">3G</option>
                                <option value="5G">5G</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="transaction_status">✅ Status</label>
                            <select id="transaction_status" name="transaction_status" required>
                                <option value="SUCCESS" selected>SUCCESS</option>
                                <option value="PENDING">PENDING</option>
                                <option value="FAILED">FAILED</option>
                            </select>
                        </div>
                    </div>
                    
                    <button type="submit" id="analyzeBtn">🔍 Analyze Transaction</button>
                </form>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Processing with Quantum AI...</p>
                </div>
                
                <div id="results" style="display: none;"></div>
            </div>
            
            <div class="panel">
                <h2>⛓️ Blockchain Network</h2>
                
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
                        <div class="stat-label">Pending</div>
                    </div>
                </div>
                
                <div class="mining-controls">
                    <button class="mine-button" onclick="mineBlock()">⛏️ Mine New Block</button>
                    <div id="miningStatus"></div>
                </div>
                
                <h3 style="margin: 20px 0 10px 0; color: #4ecdc4;">📊 Recent Transactions</h3>
                <div class="transaction-list" id="transactionList">
                    <p style="text-align: center; opacity: 0.7;">Loading transactions...</p>
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
                    document.getElementById('results').innerHTML = '<p style="color: #ff6b6b;">Error analyzing transaction</p>';
                    document.getElementById('results').style.display = 'block';
                } finally {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('analyzeBtn').disabled = false;
                }
            });
            
            function displayResults(result) {
                const riskClass = result.risk_level.includes('HIGH') ? 'risk-high' : 
                                result.risk_level.includes('MEDIUM') ? 'risk-medium' : 'risk-low';
                
                const html = `
                    <h3>🎯 Analysis Results</h3>
                    <div class="transaction-hash">Transaction Hash: ${result.transaction_hash}</div>
                    
                    <div class="score-grid">
                        <div class="score-card">
                            <div class="score-value" style="color: #4ecdc4;">${result.quantum_score}%</div>
                            <div class="score-label">Quantum AI</div>
                        </div>
                        <div class="score-card">
                            <div class="score-value" style="color: #ffa726;">${result.classical_score}%</div>
                            <div class="score-label">Classical ML</div>
                        </div>
                        <div class="score-card">
                            <div class="score-value" style="color: #ab47bc;">${result.logical_score}%</div>
                            <div class="score-label">Rule-based</div>
                        </div>
                        <div class="score-card">
                            <div class="score-value ${riskClass}" style="font-size: 2.2em;">${result.fusion_score}%</div>
                            <div class="score-label">Final Score</div>
                        </div>
                    </div>
                    
                    <div style="text-align: center; margin: 20px 0;">
                        <div style="font-size: 1.4em; font-weight: bold;" class="${riskClass}">
                            ${result.risk_level}
                        </div>
                        <div style="margin-top: 5px; opacity: 0.8;">
                            Confidence: ${result.confidence}
                        </div>
                    </div>
                    
                    <div class="success-message">
                        ✅ Transaction saved to database and ready for blockchain
                    </div>
                `;
                
                document.getElementById('results').innerHTML = html;
                document.getElementById('results').style.display = 'block';
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
                        const riskClass = tx.risk_level && tx.risk_level.includes('HIGH') ? 'risk-high' : 
                                        tx.risk_level && tx.risk_level.includes('MEDIUM') ? 'risk-medium' : 'risk-low';
                        
                        return `
                            <div class="transaction-item ${riskClass}">
                                <div class="transaction-hash">${tx.transaction_hash.substring(0, 16)}...</div>
                                <div class="transaction-details">
                                    <span>₹${tx.amount.toLocaleString()}</span>
                                    <span>${tx.risk_level || 'Pending'}</span>
                                    <span>${tx.fusion_score ? tx.fusion_score.toFixed(1) + '%' : '-'}</span>
                                </div>
                            </div>
                        `;
                    }).join('');
                    
                    document.getElementById('transactionList').innerHTML = html;
                } catch (error) {
                    console.error('Error loading transactions:', error);
                    document.getElementById('transactionList').innerHTML = '<p style="text-align: center; opacity: 0.7;">Error loading transactions</p>';
                }
            }
            
            async function mineBlock() {
                try {
                    document.getElementById('miningStatus').innerHTML = '<div class="loading" style="display: block;"><div class="spinner"></div><p>Mining block...</p></div>';
                    
                    const response = await fetch('/blockchain/mine', { method: 'POST' });
                    const result = await response.json();
                    
                    document.getElementById('miningStatus').innerHTML = `
                        <div class="success-message" style="margin-top: 10px;">
                            ⛏️ Block mined successfully!<br>
                            <small>Hash: ${result.block_hash.substring(0, 16)}...</small>
                        </div>
                    `;
                    
                    // Refresh stats and transactions
                    loadBlockchainStats();
                    loadRecentTransactions();
                    
                    // Clear mining status after 3 seconds
                    setTimeout(() => {
                        document.getElementById('miningStatus').innerHTML = '';
                    }, 3000);
                    
                } catch (error) {
                    console.error('Error mining block:', error);
                    document.getElementById('miningStatus').innerHTML = '<p style="color: #ff6b6b; text-align: center;">Error mining block</p>';
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
    """Predict fraud probability and save to database"""
    return predict_fraud_enhanced(transaction)

@app.get("/blockchain/stats")
async def get_blockchain_stats():
    """Get blockchain statistics"""
    return db.get_blockchain_info()

@app.get("/transactions/recent")
async def get_recent_transactions():
    """Get recent transactions"""
    return db.get_recent_transactions(20)

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
    """Health check"""
    return {
        "status": "healthy",
        "models_loaded": models is not None,
        "quantum_enabled": True,
        "blockchain_enabled": True,
        "database_connected": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)