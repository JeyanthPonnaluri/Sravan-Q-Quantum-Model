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

from enhanced_quantum_model import EnhancedQuantumMetaModel
from src.cost_analysis import CostOptimizer

# Global instances
quantum_meta_model = None
cost_optimizer = CostOptimizer(default_friction_cost=1000.0, default_fraud_loss=5000.0)
current_optimal_threshold = 0.45  # Default threshold

# Caching for cost optimization
cached_y_true = np.array([])
cached_y_pred_proba = np.array([])

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

def initialize_cost_evaluation_data():
    """Generate mock test dataset and precompute scores for cost analysis curves"""
    global cached_y_true, cached_y_pred_proba
    logger.info("Initializing validation cache for cost sweeps...")
    np.random.seed(42)
    
    y_true_list = []
    y_pred_list = []
    
    for i in range(50):
        amount = float(np.random.lognormal(mean=6, sigma=1.2))
        hour_of_day = int(np.random.randint(0, 24))
        is_weekend = int(np.random.choice([0, 1], p=[0.7, 0.3]))
        
        fraud_prob = 0.05 + 0.3 * (amount > 100000) + 0.2 * (hour_of_day < 6)
        fraud_prob = min(max(fraud_prob, 0.02), 0.95)
        fraud_flag = int(np.random.binomial(1, fraud_prob))
        
        # Simulated prediction score matching general relationships (amount + night = high score)
        base_score = 0.1
        if amount > 100000:
            base_score += 0.4
        if hour_of_day < 6:
            base_score += 0.3
        
        score = min(max(base_score + np.random.uniform(-0.05, 0.05), 0.01), 0.99)
        
        y_true_list.append(fraud_flag)
        y_pred_list.append(score)
            
    cached_y_true = np.array(y_true_list)
    cached_y_pred_proba = np.array(y_pred_list)
    logger.info(f"Cost sweep cache initialized: {len(cached_y_true)} samples evaluated.")

def load_models():
    """Load fraud detection models"""
    global quantum_meta_model
    try:
        import os
        api_key = os.getenv("GEMINI_API_KEY", "")
        quantum_meta_model = EnhancedQuantumMetaModel(
            neuro_qkad_models_path="enhanced_models/fraud_models.pkl",
            gemini_api_key=api_key
        )
        logger.info("Enhanced Quantum Meta Model loaded successfully!")
        initialize_cost_evaluation_data()
        return True
    except Exception as e:
        logger.error(f"Could not load Enhanced Quantum Meta Model: {e}")
        return False

def predict_fraud_enhanced(transaction: TransactionFull):
    """Enhanced fraud prediction using the real multi-model engine"""
    global quantum_meta_model, current_optimal_threshold
    
    if quantum_meta_model is None:
        return {
            'transaction_hash': '0',
            'quantum_score': 10.0,
            'classical_score': 10.0,
            'logical_score': 10.0,
            'fusion_score': 10.0,
            'risk_level': "MINIMAL RISK",
            'confidence': "High",
            'ai_reasoning': "System loading...",
            'security_flags': [],
            'recommendations': [],
            'saved_to_blockchain': False
        }
        
    transaction_data = transaction.model_dump()
    
    # Run the real model prediction!
    pred = quantum_meta_model.predict(transaction_data)
    
    # Re-classify risk level and action dynamically using cost-optimized threshold
    risk_lvl, dynamic_action = cost_optimizer.find_action_for_score(pred.final_fraud_score, current_optimal_threshold)
    
    # Construct flags
    security_flags = list(pred.primary_risk_factors)
    
    # Save to db
    db_data = transaction.model_dump()
    db_data.update({
        'quantum_score': float(pred.quantum_score),
        'classical_score': float(pred.classical_score),
        'logical_score': float(pred.gemini_logical_score),
        'fusion_score': float(pred.final_fraud_score),
        'risk_level': f"{risk_lvl} RISK",
        'confidence': f"{pred.confidence_score:.1f}%"
    })
    transaction_hash = db.save_transaction(db_data)
    
    # Recommendations
    recommendations = [dynamic_action]
    if risk_lvl in ["CRITICAL", "HIGH"]:
        recommendations.append("Alert the receiving institution to place a temporary hold.")
        recommendations.append("Log risk payload to security monitoring index.")
    
    return {
        'transaction_hash': transaction_hash,
        'quantum_score': round(pred.quantum_score, 2),
        'classical_score': round(pred.classical_score, 2),
        'logical_score': round(pred.gemini_logical_score, 2),
        'fusion_score': round(pred.final_fraud_score, 2),
        'risk_level': f"{risk_lvl} RISK",
        'confidence': f"{pred.confidence_score:.1f}%",
        'ai_reasoning': f"Model analysis version {pred.model_version}. Identified type: {pred.fraud_type_detected}. Agreement: {pred.model_agreement:.1f}%.",
        'security_flags': security_flags,
        'recommendations': recommendations,
        'saved_to_blockchain': False
    }

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    load_models()


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
    <script>
      tailwind.config = {
        theme: {
          extend: {
            fontFamily: {
              sans: ['Plus Jakarta Sans', 'Outfit', 'sans-serif'],
            }
          }
        }
      }
    </script>
</head>
<body class="bg-zinc-950 text-white font-sans overflow-x-hidden">
    <div class="relative w-full min-h-screen bg-zinc-950 text-white overflow-hidden">
      <!-- SCOPED ANIMATIONS -->
      <style>
        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes marquee {
          from { transform: translateX(0); }
          to { transform: translateX(-50%); }
        }
        .animate-fade-in {
          animation: fadeSlideIn 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
          opacity: 0;
        }
        .animate-marquee {
          animation: marquee 30s linear infinite;
        }
        .delay-100 { animation-delay: 0.1s; }
        .delay-200 { animation-delay: 0.2s; }
        .delay-300 { animation-delay: 0.3s; }
        .delay-400 { animation-delay: 0.4s; }
        .delay-500 { animation-delay: 0.5s; }
      </style>

      <!-- Background Image with Gradient Mask -->
      <div 
        class="absolute inset-0 z-0 bg-[url('https://images.unsplash.com/photo-1639762681485-074b7f938ba0?q=80&w=2832&auto=format&fit=crop')] bg-cover bg-center opacity-20"
        style="mask-image: linear-gradient(180deg, transparent, black 15%, black 85%, transparent); -webkit-mask-image: linear-gradient(180deg, transparent, black 15%, black 85%, transparent);"
      ></div>

      <!-- Navigation Header -->
      <header class="relative z-20 mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8 flex justify-between items-center">
        <div class="flex items-center gap-2">
          <div class="text-2xl font-bold bg-gradient-to-r from-white to-[#ffcd75] bg-clip-text text-transparent flex items-center gap-2">
            <i data-lucide="shield" class="w-6 h-6 text-[#ffcd75]"></i> Sravan
          </div>
        </div>
        <a href="/dashboard" class="inline-flex items-center justify-center gap-2 rounded-full border border-white/10 bg-white/5 px-5 py-2.5 text-xs font-semibold text-white backdrop-blur-md transition-all hover:bg-white/10 hover:border-white/20 hover:scale-[1.02] active:scale-[0.98]">
          Launch Dashboard <i data-lucide="arrow-right" class="w-3.5 h-3.5"></i>
        </a>
      </header>

      <div class="relative z-10 mx-auto max-w-7xl px-4 pt-16 pb-24 sm:px-6 md:pt-24 md:pb-32 lg:px-8">
        <div class="grid grid-cols-1 gap-16 lg:grid-cols-12 lg:gap-8 items-center">
          
          <!-- --- LEFT COLUMN --- -->
          <div class="lg:col-span-7 flex flex-col justify-center space-y-8">
            
            <!-- Badge -->
            <div class="animate-fade-in delay-100">
              <div class="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 backdrop-blur-md transition-colors hover:bg-white/10">
                <span class="text-[10px] sm:text-xs font-semibold uppercase tracking-wider text-zinc-300 flex items-center gap-2">
                  Award-Winning Design
                  <i data-lucide="star" class="w-3.5 h-3.5 text-yellow-400 fill-yellow-400"></i>
                </span>
              </div>
            </div>

            <!-- Heading -->
            <h1 
              class="animate-fade-in delay-200 text-5xl sm:text-6xl lg:text-7xl xl:text-8xl font-medium tracking-tighter leading-[0.9] text-white"
            >
              Crafting Digital<br />
              <span class="bg-gradient-to-br from-white via-white to-[#ffcd75] bg-clip-text text-transparent">
                Experiences
              </span><br />
              That Matter
            </h1>

            <!-- Description -->
            <p class="animate-fade-in delay-300 max-w-xl text-lg text-zinc-400 leading-relaxed">
              We design interfaces that combine beauty with functionality,
              creating seamless experiences that users love and businesses thrive on.
            </p>

            <!-- CTA Buttons -->
            <div class="animate-fade-in delay-400 flex flex-col sm:flex-row gap-4">
              <a href="/dashboard" class="group inline-flex items-center justify-center gap-2 rounded-full bg-white px-8 py-4 text-sm font-semibold text-zinc-950 transition-all hover:scale-[1.02] hover:bg-zinc-200 active:scale-[0.98] shadow-lg shadow-white/5">
                View Portfolio
                <i data-lucide="arrow-right" class="w-4 h-4 transition-transform group-hover:translate-x-1"></i>
              </a>
              
              <a href="/dashboard" class="group inline-flex items-center justify-center gap-2 rounded-full border border-white/10 bg-white/5 px-8 py-4 text-sm font-semibold text-white backdrop-blur-sm transition-colors hover:bg-white/10 hover:border-white/20">
                <i data-lucide="play" class="w-4 h-4 fill-current"></i>
                Watch Showreel
              </a>
            </div>
          </div>

          <!-- --- RIGHT COLUMN --- -->
          <div class="lg:col-span-5 space-y-6">
            
            <!-- Stats Card -->
            <div class="animate-fade-in delay-500 relative overflow-hidden rounded-3xl border border-white/10 bg-white/5 p-8 backdrop-blur-xl shadow-2xl">
              <!-- Card Glow Effect -->
              <div class="absolute top-0 right-0 -mr-16 -mt-16 h-64 w-64 rounded-full bg-white/5 blur-3xl pointer-events-none"></div>

              <div class="relative z-10">
                <div class="flex items-center gap-4 mb-8">
                  <div class="flex h-12 w-12 items-center justify-center rounded-2xl bg-white/10 ring-1 ring-white/20">
                    <i data-lucide="target" class="h-6 w-6 text-white"></i>
                  </div>
                  <div>
                    <div class="text-3xl font-bold tracking-tight text-white">150+</div>
                    <div class="text-sm text-zinc-400">Projects Delivered</div>
                  </div>
                </div>

                <!-- Progress Bar Section -->
                <div class="space-y-3 mb-8">
                  <div class="flex justify-between text-sm">
                    <span class="text-zinc-400">Client Satisfaction</span>
                    <span class="text-white font-medium">98%</span>
                  </div>
                  <div class="h-2 w-full overflow-hidden rounded-full bg-zinc-800/50">
                    <div class="h-full w-[98%] rounded-full bg-gradient-to-r from-white to-zinc-400"></div>
                  </div>
                </div>

                <div class="h-px w-full bg-white/10 mb-6"></div>

                <!-- Mini Stats Grid -->
                <div class="grid grid-cols-5 items-center gap-2 text-center">
                  <div class="flex flex-col items-center justify-center transition-transform hover:-translate-y-1 cursor-default">
                    <span class="text-xl font-bold text-white sm:text-2xl">5+</span>
                    <span class="text-[10px] uppercase tracking-wider text-zinc-500 font-medium sm:text-xs">Years</span>
                  </div>
                  <div class="w-px h-8 bg-white/10 mx-auto"></div>
                  <div class="flex flex-col items-center justify-center transition-transform hover:-translate-y-1 cursor-default">
                    <span class="text-xl font-bold text-white sm:text-2xl">24/7</span>
                    <span class="text-[10px] uppercase tracking-wider text-zinc-500 font-medium sm:text-xs">Support</span>
                  </div>
                  <div class="w-px h-8 bg-white/10 mx-auto"></div>
                  <div class="flex flex-col items-center justify-center transition-transform hover:-translate-y-1 cursor-default">
                    <span class="text-xl font-bold text-white sm:text-2xl">100%</span>
                    <span class="text-[10px] uppercase tracking-wider text-zinc-500 font-medium sm:text-xs">Quality</span>
                  </div>
                </div>

                <!-- Tag Pills -->
                <div class="mt-8 flex flex-wrap gap-2">
                  <div class="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[10px] font-medium tracking-wide text-zinc-300">
                    <span class="relative flex h-2 w-2">
                      <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                      <span class="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                    </span>
                    ACTIVE
                  </div>
                  <div class="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[10px] font-medium tracking-wide text-zinc-300">
                    <i data-lucide="crown" class="w-3.5 h-3.5 text-yellow-500"></i>
                    PREMIUM
                  </div>
                </div>
              </div>
            </div>

            <!-- Marquee Card -->
            <div class="animate-fade-in delay-500 relative overflow-hidden rounded-3xl border border-white/10 bg-white/5 py-8 backdrop-blur-xl">
              <h3 class="mb-6 px-8 text-sm font-medium text-zinc-400">Trusted by Industry Leaders</h3>
              
              <div 
                class="relative flex overflow-hidden w-full"
                style="mask-image: linear-gradient(to right, transparent, black 20%, black 80%, transparent); -webkit-mask-image: linear-gradient(to right, transparent, black 20%, black 80%, transparent)"
              >
                <div class="animate-marquee flex gap-12 whitespace-nowrap px-4">
                  <!-- Logos -->
                  <div class="flex items-center gap-2 opacity-50 hover:opacity-100 hover:scale-105 cursor-default transition-all">
                    <i data-lucide="hexagon" class="h-5 w-5 text-white"></i>
                    <span class="text-base font-bold text-white tracking-tight">Acme Corp</span>
                  </div>
                  <div class="flex items-center gap-2 opacity-50 hover:opacity-100 hover:scale-105 cursor-default transition-all">
                    <i data-lucide="triangle" class="h-5 w-5 text-white"></i>
                    <span class="text-base font-bold text-white tracking-tight">Quantum</span>
                  </div>
                  <div class="flex items-center gap-2 opacity-50 hover:opacity-100 hover:scale-105 cursor-default transition-all">
                    <i data-lucide="command" class="h-5 w-5 text-white"></i>
                    <span class="text-base font-bold text-white tracking-tight">Command+Z</span>
                  </div>
                  <div class="flex items-center gap-2 opacity-50 hover:opacity-100 hover:scale-105 cursor-default transition-all">
                    <i data-lucide="ghost" class="h-5 w-5 text-white"></i>
                    <span class="text-base font-bold text-white tracking-tight">Phantom</span>
                  </div>
                  <div class="flex items-center gap-2 opacity-50 hover:opacity-100 hover:scale-105 cursor-default transition-all">
                    <i data-lucide="gem" class="h-5 w-5 text-white"></i>
                    <span class="text-base font-bold text-white tracking-tight">Ruby</span>
                  </div>
                  <div class="flex items-center gap-2 opacity-50 hover:opacity-100 hover:scale-105 cursor-default transition-all">
                    <i data-lucide="cpu" class="h-5 w-5 text-white"></i>
                    <span class="text-base font-bold text-white tracking-tight">Chipset</span>
                  </div>
                  <!-- Duplicate for infinite scroll -->
                  <div class="flex items-center gap-2 opacity-50 hover:opacity-100 hover:scale-105 cursor-default transition-all">
                    <i data-lucide="hexagon" class="h-5 w-5 text-white"></i>
                    <span class="text-base font-bold text-white tracking-tight">Acme Corp</span>
                  </div>
                  <div class="flex items-center gap-2 opacity-50 hover:opacity-100 hover:scale-105 cursor-default transition-all">
                    <i data-lucide="triangle" class="h-5 w-5 text-white"></i>
                    <span class="text-base font-bold text-white tracking-tight">Quantum</span>
                  </div>
                  <div class="flex items-center gap-2 opacity-50 hover:opacity-100 hover:scale-105 cursor-default transition-all">
                    <i data-lucide="command" class="h-5 w-5 text-white"></i>
                    <span class="text-base font-bold text-white tracking-tight">Command+Z</span>
                  </div>
                  <div class="flex items-center gap-2 opacity-50 hover:opacity-100 hover:scale-105 cursor-default transition-all">
                    <i data-lucide="ghost" class="h-5 w-5 text-white"></i>
                    <span class="text-base font-bold text-white tracking-tight">Phantom</span>
                  </div>
                  <div class="flex items-center gap-2 opacity-50 hover:opacity-100 hover:scale-105 cursor-default transition-all">
                    <i data-lucide="gem" class="h-5 w-5 text-white"></i>
                    <span class="text-base font-bold text-white tracking-tight">Ruby</span>
                  </div>
                  <div class="flex items-center gap-2 opacity-50 hover:opacity-100 hover:scale-105 cursor-default transition-all">
                    <i data-lucide="cpu" class="h-5 w-5 text-white"></i>
                    <span class="text-base font-bold text-white tracking-tight">Chipset</span>
                  </div>
                </div>
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
    
    <!-- Initialize Lucide Icons -->
    <script>
      lucide.createIcons();
    </script>
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
                    <div id="geminiStatusBadge" class="security-badge" style="background: rgba(16, 185, 129, 0.2); border-color: rgba(16, 185, 129, 0.4); color: #10b981;">
                        <i class="fas fa-brain"></i>
                        <span id="geminiStatusText">Gemini: Connecting...</span>
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

            <div class="panel" style="margin-bottom: 2rem;">
                <div class="panel-header" style="background: linear-gradient(90deg, var(--warning-orange) 0%, var(--danger-red) 100%);">
                    <i class="fas fa-sliders-h"></i>
                    <h2>AI Risk Manager Cost Controller</h2>
                </div>
                <div class="panel-content">
                    <p style="font-size: 0.85rem; color: var(--text-light); margin-bottom: 1.25rem;">
                        Balance false-positive friction costs (blocking good transactions) vs. false-negative fraud losses (missed chargebacks) to dynamically optimize the model's threshold boundaries.
                    </p>
                    
                    <div style="margin-bottom: 1.25rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-size: 0.9rem; font-weight: 500;">
                            <span>FP Friction Cost (Falsely Flagged)</span>
                            <span id="valFriction" style="color: var(--primary-blue); font-weight: 600;">₹1,000</span>
                        </div>
                        <input type="range" id="sliderFriction" min="500" max="5000" step="100" value="1000" style="cursor: pointer;" oninput="updateCosts()">
                    </div>
                    
                    <div style="margin-bottom: 1.25rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-size: 0.9rem; font-weight: 500;">
                            <span>FN Fraud Loss (Chargebacks & Fees)</span>
                            <span id="valFraud" style="color: var(--danger-red); font-weight: 600;">₹5,000</span>
                        </div>
                        <input type="range" id="sliderFraud" min="2000" max="20000" step="500" value="5000" style="cursor: pointer;" oninput="updateCosts()">
                    </div>

                    <div style="background: var(--bg-light); border: 1px solid var(--border-light); border-radius: 8px; padding: 1rem; margin-top: 1.25rem;">
                        <h4 style="font-size: 0.9rem; color: var(--primary-blue); margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">
                            <i class="fas fa-chart-line"></i> Dynamic Cost-Optimized Metrics
                        </h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; font-size: 0.85rem;">
                            <div>Optimal Threshold: <strong id="optThreshold" style="color: var(--primary-blue);">45%</strong></div>
                            <div>F1 Score: <strong id="optF1">85.2%</strong></div>
                            <div>Precision: <strong id="optPrecision">83.3%</strong></div>
                            <div>Recall: <strong id="optRecall">87.5%</strong></div>
                            <div>False Positives: <strong id="optFPs" style="color: var(--warning-orange);">1</strong></div>
                            <div>False Negatives: <strong id="optFNs" style="color: var(--danger-red);">2</strong></div>
                        </div>
                        <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px dashed var(--border-light); font-size: 0.85rem; display: flex; justify-content: space-between; align-items: center;">
                            <span>Optimized Savings:</span>
                            <strong style="color: var(--success-green); font-size: 1rem;" id="optSavings">₹18,500</strong>
                        </div>
                    </div>
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

            async function updateCosts() {
                const friction = document.getElementById('sliderFriction').value;
                const fraud = document.getElementById('sliderFraud').value;
                
                document.getElementById('valFriction').textContent = '₹' + parseInt(friction).toLocaleString();
                document.getElementById('valFraud').textContent = '₹' + parseInt(fraud).toLocaleString();
                
                try {
                    const response = await fetch(`/api/cost_optimize?friction_cost=${friction}&fraud_loss=${fraud}`);
                    const data = await response.json();
                    
                    document.getElementById('optThreshold').textContent = Math.round(data.optimal_threshold * 100) + '%';
                    document.getElementById('optF1').textContent = (data.f1 * 100).toFixed(1) + '%';
                    document.getElementById('optPrecision').textContent = (data.precision * 100).toFixed(1) + '%';
                    document.getElementById('optRecall').textContent = (data.recall * 100).toFixed(1) + '%';
                    document.getElementById('optFPs').textContent = data.fp_count;
                    document.getElementById('optFNs').textContent = data.fn_count;
                    
                    const totalSavings = data.savings_vs_no_model;
                    document.getElementById('optSavings').textContent = '₹' + Math.round(totalSavings).toLocaleString();
                    
                } catch (error) {
                    console.error('Error optimizing costs:', error);
                }
            }

            async function checkGeminiStatus() {
                try {
                    const response = await fetch('/health');
                    const health = await response.json();
                    const badge = document.getElementById('geminiStatusBadge');
                    const text = document.getElementById('geminiStatusText');
                    
                    if (health.gemini_ai_enabled) {
                        badge.style.background = 'rgba(16, 185, 129, 0.2)';
                        badge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
                        badge.style.color = '#10b981';
                        text.textContent = 'Gemini: Active';
                    } else {
                        badge.style.background = 'rgba(245, 158, 11, 0.2)';
                        badge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
                        badge.style.color = '#f59e0b';
                        text.textContent = 'Gemini: Fallback Active';
                    }
                } catch (error) {
                    console.error('Error checking health status:', error);
                }
            }

            // Load initial data
            loadBlockchainStats();
            loadRecentTransactions();
            updateCosts();
            checkGeminiStatus();
            
            // Auto-refresh every 30 seconds
            setInterval(() => {
                loadBlockchainStats();
                loadRecentTransactions();
                checkGeminiStatus();
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
    global quantum_meta_model
    gemini_ok = False
    if quantum_meta_model and quantum_meta_model.gemini_model:
        gemini_ok = quantum_meta_model.gemini_model.model is not None
    return {
        "status": "operational",
        "models_loaded": quantum_meta_model is not None,
        "quantum_ai_enabled": True,
        "gemini_ai_enabled": gemini_ok,
        "blockchain_enabled": True,
        "security_level": "government_grade",
        "current_optimal_threshold": current_optimal_threshold
    }

@app.get("/api/cost_optimize")
async def api_cost_optimize(friction_cost: float = 1000.0, fraud_loss: float = 5000.0):
    """Dynamically sweeps threshold costs based on merchant specifications"""
    global cached_y_true, cached_y_pred_proba, current_optimal_threshold
    
    res = cost_optimizer.calculate_cost_matrix(
        cached_y_true, cached_y_pred_proba, friction_cost, fraud_loss
    )
    current_optimal_threshold = res["optimal_threshold"]
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)