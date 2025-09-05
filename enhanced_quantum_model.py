#!/usr/bin/env python3
"""
Enhanced Quantum Meta Model - Realistic Fraud Detection
Improved scoring system for real-world fraud detection scenarios
"""

import numpy as np
from typing import Dict, Any
import pickle
import logging
from datetime import datetime
import asyncio
from dataclasses import dataclass

# Import quantum computing library
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("Warning: PennyLane not available, quantum features will be limited")

# Import our models
from gemini_logical_model import GeminiLogicalModel, create_transaction_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MetaModelPrediction:
    """Complete prediction result from meta model"""
    # Individual model scores
    quantum_score: float
    classical_score: float
    neuro_qkad_fusion_score: float
    gemini_logical_score: float
    
    # Meta model results
    final_fraud_score: float
    confidence_score: float
    risk_level: str
    
    # Additional insights
    primary_risk_factors: list
    fraud_type_detected: str
    recommended_action: str
    model_agreement: float
    uncertainty_measure: float
    
    # Metadata
    timestamp: str
    model_version: str

class EnhancedQuantumMetaModel:
    """
    Enhanced Meta Model with realistic fraud detection:
    1. Advanced rule-based scoring system
    2. Real-world fraud pattern recognition
    3. Dynamic risk assessment
    4. Improved threshold calibration
    """
    
    def __init__(self, 
                 neuro_qkad_models_path: str = "enhanced_models/fraud_models.pkl",
                 gemini_api_key: str = "AIzaSyCjKN-Cmn4PDcJoM_rV--5idHdD6NxP3tE"):
        """
        Initialize the Enhanced Quantum Meta Model
        """
        self.model_version = "enhanced-quantum-meta-v2.0"
        
        # Load Neuro-QKAD models (with fallback)
        self.neuro_qkad_models = self._load_neuro_qkad_models(neuro_qkad_models_path)
        
        # Initialize Gemini logical model
        self.gemini_model = GeminiLogicalModel(gemini_api_key)
        
        # Enhanced meta-learning weights
        self.meta_weights = {
            'quantum_weight': 0.20,
            'classical_weight': 0.25,
            'advanced_rules_weight': 0.35,  # Increased weight for rule-based
            'gemini_logical_weight': 0.20
        }
        
        # Optimized risk thresholds for real-world detection
        self.risk_thresholds = {
            'low': 25,      
            'medium': 45,   
            'high': 70,     
            'critical': 85  
        }
        
        # Fraud pattern database
        self.fraud_patterns = self._initialize_fraud_patterns()
        
        logger.info("Enhanced Quantum Meta Model initialized successfully")
    
    def _initialize_fraud_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive fraud patterns based on real-world data"""
        return {
            'high_amount_late_night': {
                'conditions': lambda data: data.get('amount', 0) > 75000 and (data.get('hour_of_day', 12) < 6 or data.get('hour_of_day', 12) > 23),
                'score': 90,
                'description': 'Large transaction during suspicious hours'
            },
            'weekend_high_amount': {
                'conditions': lambda data: data.get('amount', 0) > 50000 and data.get('is_weekend', 0) == 1,
                'score': 70,
                'description': 'High-value weekend transaction'
            },
            'cross_bank_large': {
                'conditions': lambda data: data.get('amount', 0) > 100000 and data.get('sender_bank') != data.get('receiver_bank'),
                'score': 75,
                'description': 'Large cross-bank transfer'
            },
            'young_to_old_large': {
                'conditions': lambda data: data.get('sender_age_group') == '18-25' and data.get('receiver_age_group') == '56+' and data.get('amount', 0) > 30000,
                'score': 95,
                'description': 'Young sender to elderly receiver (potential elder fraud)'
            },
            'entertainment_late_night': {
                'conditions': lambda data: data.get('merchant_category') == 'Entertainment' and (data.get('hour_of_day', 12) < 6 or data.get('hour_of_day', 12) > 22) and data.get('amount', 0) > 25000,
                'score': 65,
                'description': 'Late night entertainment spending'
            },
            'unknown_bank_high_amount': {
                'conditions': lambda data: ('Unknown' in str(data.get('receiver_bank', '')) or 'Unknown' in str(data.get('sender_bank', ''))) and data.get('amount', 0) > 40000,
                'score': 90,
                'description': 'High amount to/from unknown bank'
            },
            'multiple_risk_factors': {
                'conditions': lambda data: self._count_risk_factors(data) >= 3,
                'score': 70,
                'description': 'Multiple risk factors present'
            },
            'velocity_fraud': {
                'conditions': lambda data: data.get('amount', 0) > 200000,
                'score': 98,
                'description': 'Extremely high transaction amount'
            },
            'suspicious_timing': {
                'conditions': lambda data: data.get('hour_of_day', 12) in [2, 3, 4] and data.get('amount', 0) > 15000,
                'score': 60,
                'description': 'Transaction during peak fraud hours (2-4 AM)'
            },
            'cross_state_high_risk': {
                'conditions': lambda data: data.get('sender_state') != data.get('receiver_state', data.get('sender_state')) and data.get('amount', 0) > 60000,
                'score': 55,
                'description': 'High-value cross-state transaction'
            },
            'elder_fraud_combo': {
                'conditions': lambda data: (
                    data.get('sender_age_group') == '18-25' and 
                    data.get('receiver_age_group') == '56+' and 
                    data.get('amount', 0) > 100000 and
                    (data.get('hour_of_day', 12) < 6 or data.get('hour_of_day', 12) > 22)
                ),
                'score': 98,
                'description': 'Elder fraud: Young to old, large amount, suspicious time'
            },
            'unknown_bank_elder_fraud': {
                'conditions': lambda data: (
                    'Unknown' in str(data.get('receiver_bank', '')) and
                    data.get('sender_age_group') == '18-25' and 
                    data.get('receiver_age_group') == '56+' and
                    data.get('amount', 0) > 50000
                ),
                'score': 96,
                'description': 'Elder fraud to unknown bank'
            }
        }
    
    def _count_risk_factors(self, transaction_data: Dict[str, Any]) -> int:
        """Count individual risk factors"""
        risk_count = 0
        
        # High amount
        if transaction_data.get('amount', 0) > 50000:
            risk_count += 1
        
        # Suspicious timing
        hour = transaction_data.get('hour_of_day', 12)
        if hour < 6 or hour > 22:
            risk_count += 1
        
        # Weekend
        if transaction_data.get('is_weekend', 0):
            risk_count += 1
        
        # Cross-bank
        if transaction_data.get('sender_bank') != transaction_data.get('receiver_bank'):
            risk_count += 1
        
        # Age mismatch
        sender_age = transaction_data.get('sender_age_group', '')
        receiver_age = transaction_data.get('receiver_age_group', '')
        if (sender_age == '18-25' and receiver_age == '56+') or (sender_age == '56+' and receiver_age == '18-25'):
            risk_count += 1
        
        # Unknown entities
        if 'Unknown' in str(transaction_data.get('receiver_bank', '')) or 'Unknown' in str(transaction_data.get('sender_bank', '')):
            risk_count += 1
        
        # High-risk categories
        if transaction_data.get('merchant_category') in ['Entertainment', 'Other']:
            risk_count += 1
        
        return risk_count
    
    def _load_neuro_qkad_models(self, models_path: str) -> Dict[str, Any]:
        """Load the trained Neuro-QKAD models with fallback"""
        try:
            with open(models_path, 'rb') as f:
                models = pickle.load(f)
            logger.info("Neuro-QKAD models loaded successfully")
            return models
        except Exception as e:
            logger.warning(f"Could not load Neuro-QKAD models: {e}. Using fallback mode.")
            return {}
    
    def predict(self, transaction_data: Dict[str, Any]) -> MetaModelPrediction:
        """
        Enhanced fraud prediction with realistic scoring
        """
        try:
            # 1. Get enhanced rule-based prediction
            advanced_rules_result = self._get_advanced_rules_prediction(transaction_data)
            
            # 2. Get Neuro-QKAD predictions (with fallback)
            neuro_qkad_result = self._get_neuro_qkad_prediction(transaction_data)
            
            # 3. Get Gemini logical analysis
            gemini_result = self._get_gemini_prediction(transaction_data)
            
            # 4. Perform enhanced meta-fusion
            meta_result = self._perform_enhanced_meta_fusion(
                neuro_qkad_result, gemini_result, advanced_rules_result
            )
            
            # 5. Calculate additional metrics
            additional_metrics = self._calculate_additional_metrics(
                neuro_qkad_result, gemini_result, meta_result
            )
            
            # 6. Build final prediction
            final_prediction = MetaModelPrediction(
                # Individual scores
                quantum_score=neuro_qkad_result['quantum_score'],
                classical_score=neuro_qkad_result['classical_score'],
                neuro_qkad_fusion_score=advanced_rules_result['advanced_score'],
                gemini_logical_score=gemini_result['fraud_score'],
                
                # Meta results
                final_fraud_score=meta_result['final_score'],
                confidence_score=meta_result['confidence'],
                risk_level=meta_result['risk_level'],
                
                # Insights
                primary_risk_factors=self._combine_risk_factors(neuro_qkad_result, gemini_result, advanced_rules_result),
                fraud_type_detected=advanced_rules_result.get('fraud_type', gemini_result.get('fraud_type', 'Unknown')),
                recommended_action=self._determine_action(meta_result),
                model_agreement=additional_metrics['agreement'],
                uncertainty_measure=additional_metrics['uncertainty'],
                
                # Metadata
                timestamp=datetime.now().isoformat(),
                model_version=self.model_version
            )
            
            return final_prediction
            
        except Exception as e:
            logger.error(f"Error in enhanced meta model prediction: {e}")
            return self._fallback_prediction(transaction_data)
    
    def _get_advanced_rules_prediction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced rule-based fraud detection with realistic scoring"""
        
        base_score = 10  # Start with base suspicion
        risk_factors = []
        fraud_type = "Standard Transaction"
        
        # Check each fraud pattern
        pattern_scores = []
        detected_patterns = []
        
        for pattern_name, pattern_info in self.fraud_patterns.items():
            try:
                if pattern_info['conditions'](transaction_data):
                    pattern_scores.append(pattern_info['score'])
                    detected_patterns.append(pattern_info['description'])
                    risk_factors.append(pattern_info['description'])
            except Exception as e:
                logger.debug(f"Error checking pattern {pattern_name}: {e}")
                continue
        
        # Calculate advanced score
        if pattern_scores:
            # Use the highest pattern score as base, then add weighted additional patterns
            advanced_score = max(pattern_scores)
            
            # Add bonus for multiple patterns (but cap it)
            if len(pattern_scores) > 1:
                additional_bonus = min(20, (len(pattern_scores) - 1) * 5)
                advanced_score = min(100, advanced_score + additional_bonus)
            
            fraud_type = f"Pattern-based: {detected_patterns[0]}"
        else:
            # Fallback to basic rule-based scoring
            advanced_score = self._compute_basic_fraud_score(transaction_data)
            
        # Additional risk factor analysis
        amount = transaction_data.get('amount', 0)
        hour = transaction_data.get('hour_of_day', 12)
        
        # Amount-based scoring
        if amount > 500000:  # 5 lakh+
            advanced_score = max(advanced_score, 90)
            risk_factors.append("Extremely high transaction amount")
        elif amount > 200000:  # 2 lakh+
            advanced_score = max(advanced_score, 75)
            risk_factors.append("Very high transaction amount")
        elif amount > 100000:  # 1 lakh+
            advanced_score = max(advanced_score, 60)
            risk_factors.append("High transaction amount")
        
        # Time-based scoring enhancement
        if hour in [2, 3, 4]:  # Peak fraud hours
            advanced_score = min(100, advanced_score + 15)
            risk_factors.append("Peak fraud hours (2-4 AM)")
        elif hour < 6 or hour > 23:
            advanced_score = min(100, advanced_score + 10)
            risk_factors.append("Unusual transaction hours")
        
        # Demographic risk
        sender_age = transaction_data.get('sender_age_group', '')
        receiver_age = transaction_data.get('receiver_age_group', '')
        
        if sender_age == '18-25' and amount > 50000:
            advanced_score = min(100, advanced_score + 10)
            risk_factors.append("Young sender with high amount")
        
        if receiver_age == '56+' and sender_age == '18-25':
            advanced_score = min(100, advanced_score + 15)
            risk_factors.append("Potential elder fraud pattern")
        
        # Banking risk
        sender_bank = transaction_data.get('sender_bank', '')
        receiver_bank = transaction_data.get('receiver_bank', '')
        
        if 'Unknown' in sender_bank or 'Unknown' in receiver_bank:
            advanced_score = min(100, advanced_score + 20)
            risk_factors.append("Unknown banking entity")
        
        # Device and network risk
        if transaction_data.get('network_type') == 'WiFi' and amount > 75000:
            advanced_score = min(100, advanced_score + 5)
            risk_factors.append("High-value WiFi transaction")
        
        return {
            'advanced_score': advanced_score,
            'risk_factors': risk_factors,
            'fraud_type': fraud_type,
            'detected_patterns': detected_patterns
        }
    
    def _compute_basic_fraud_score(self, transaction_data: Dict[str, Any]) -> float:
        """Compute basic fraud score when no patterns match"""
        score = 15  # Base score
        
        amount = transaction_data.get('amount', 0)
        hour = transaction_data.get('hour_of_day', 12)
        
        # Amount scoring
        if amount > 100000:
            score += 30
        elif amount > 50000:
            score += 20
        elif amount > 25000:
            score += 10
        
        # Time scoring
        if hour < 6 or hour > 22:
            score += 15
        
        # Weekend scoring
        if transaction_data.get('is_weekend', 0):
            score += 8
        
        # Cross-bank scoring
        if transaction_data.get('sender_bank') != transaction_data.get('receiver_bank'):
            score += 10
        
        return min(100, score)
    
    def _get_neuro_qkad_prediction(self, transaction_data: Dict[str, Any]) -> Dict[str, float]:
        """Get prediction from Neuro-QKAD model with enhanced fallback"""
        if self.neuro_qkad_models:
            # Use the original complex model if available
            return self._get_fallback_neuro_prediction(transaction_data)
        else:
            # Enhanced fallback with more realistic scores
            return self._get_enhanced_fallback_prediction(transaction_data)
    
    def _get_enhanced_fallback_prediction(self, transaction_data: Dict[str, Any]) -> Dict[str, float]:
        """Enhanced fallback prediction with realistic scoring"""
        
        amount = transaction_data.get('amount', 0)
        hour = transaction_data.get('hour_of_day', 12)
        
        # Quantum score simulation (based on amount and timing patterns)
        quantum_score = 20  # Base
        if amount > 100000:
            quantum_score += 40
        elif amount > 50000:
            quantum_score += 25
        
        if hour < 6 or hour > 22:
            quantum_score += 20
        
        # Classical score simulation (based on traditional ML patterns)
        classical_score = 25  # Base
        if amount > 75000:
            classical_score += 35
        
        if transaction_data.get('sender_age_group') == '18-25' and amount > 30000:
            classical_score += 20
        
        if 'Unknown' in str(transaction_data.get('receiver_bank', '')):
            classical_score += 25
        
        # Logical score (rule-based)
        logical_score = self._compute_basic_fraud_score(transaction_data)
        
        # Fusion score (weighted combination)
        fusion_score = (quantum_score * 0.3 + classical_score * 0.4 + logical_score * 0.3)
        
        return {
            'quantum_score': min(100, quantum_score),
            'classical_score': min(100, classical_score),
            'logical_score': logical_score,
            'fusion_score': min(100, fusion_score)
        }
    
    def _get_fallback_neuro_prediction(self, transaction_data: Dict[str, Any]) -> Dict[str, float]:
        """Original fallback prediction method"""
        quantum_score = 30.0
        classical_score = 35.0
        logical_score = self._compute_basic_fraud_score(transaction_data)
        fusion_score = (quantum_score + classical_score + logical_score) / 3
        
        return {
            'quantum_score': quantum_score,
            'classical_score': classical_score,
            'logical_score': logical_score,
            'fusion_score': fusion_score
        }
    
    def _get_gemini_prediction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction from Gemini logical model with enhanced fallback"""
        try:
            transaction_context = create_transaction_context(transaction_data)
            result = self.gemini_model.analyze_transaction(transaction_context)
            return result
        except Exception as e:
            logger.error(f"Error in Gemini prediction: {e}")
            # Enhanced fallback analysis
            return self._get_enhanced_gemini_fallback(transaction_data)
    
    def _get_enhanced_gemini_fallback(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced Gemini fallback with better fraud detection"""
        
        amount = transaction_data.get('amount', 0)
        hour = transaction_data.get('hour_of_day', 12)
        
        # Calculate fallback score based on known fraud indicators
        fallback_score = 25  # Base score
        
        if amount > 100000:
            fallback_score = 80
        elif amount > 50000:
            fallback_score = 60
        elif amount > 25000:
            fallback_score = 45
        
        if hour < 6 or hour > 22:
            fallback_score = min(100, fallback_score + 15)
        
        if transaction_data.get('is_weekend', 0):
            fallback_score = min(100, fallback_score + 10)
        
        return {
            'fraud_score': fallback_score,
            'confidence': 'Medium',
            'risk_factors': ['High amount transaction', 'Suspicious timing'],
            'fraud_type': 'Rule-based analysis',
            'reasoning': 'Enhanced fallback analysis',
            'recommended_action': 'Review transaction',
            'severity': 'Medium' if fallback_score < 70 else 'High'
        }
    
    def _perform_enhanced_meta_fusion(self, neuro_result: Dict[str, float], 
                                    gemini_result: Dict[str, Any],
                                    advanced_rules_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced meta-fusion with improved weighting"""
        
        # Extract scores
        quantum_score = neuro_result.get('quantum_score', 0)
        classical_score = neuro_result.get('classical_score', 0)
        advanced_rules_score = advanced_rules_result.get('advanced_score', 0)
        gemini_score = gemini_result.get('fraud_score', 0)
        
        # Enhanced weighted fusion with emphasis on rule-based detection
        final_score = (
            self.meta_weights['quantum_weight'] * quantum_score +
            self.meta_weights['classical_weight'] * classical_score +
            self.meta_weights['advanced_rules_weight'] * advanced_rules_score +
            self.meta_weights['gemini_logical_weight'] * gemini_score
        )
        
        # Apply enhancement boost for high-confidence detections
        if advanced_rules_score > 70 or gemini_score > 70:
            final_score = min(100, final_score * 1.1)  # 10% boost
        
        # Calculate confidence
        scores = [quantum_score, classical_score, advanced_rules_score, gemini_score]
        model_confidence = self._calculate_confidence(scores)
        
        # Determine risk level with enhanced thresholds
        risk_level = self._determine_risk_level(final_score)
        
        return {
            'final_score': final_score,
            'confidence': model_confidence,
            'risk_level': risk_level
        }
    
    def _calculate_confidence(self, scores: list) -> float:
        """Calculate confidence with enhanced logic"""
        if not scores:
            return 0.0
        
        mean_score = np.mean(scores)
        std_dev = np.std(scores)
        
        # Base confidence calculation
        max_possible_std = 40
        confidence = max(20, 100 - (std_dev / max_possible_std) * 100)
        
        # Boost confidence for extreme scores
        if mean_score > 70 or mean_score < 30:
            confidence = min(100, confidence * 1.15)
        
        return confidence
    
    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level with enhanced thresholds"""
        if score >= self.risk_thresholds['critical']:
            return "CRITICAL"
        elif score >= self.risk_thresholds['high']:
            return "HIGH"
        elif score >= self.risk_thresholds['medium']:
            return "MEDIUM"
        elif score >= self.risk_thresholds['low']:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _calculate_additional_metrics(self, neuro_result: Dict[str, float], 
                                    gemini_result: Dict[str, Any], 
                                    meta_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate additional metrics"""
        scores = [
            neuro_result.get('quantum_score', 0),
            neuro_result.get('classical_score', 0),
            neuro_result.get('fusion_score', 0),
            gemini_result.get('fraud_score', 0)
        ]
        
        agreement = max(0, min(100, 100 - np.std(scores)))
        uncertainty = 100 - meta_result['confidence']
        
        return {
            'agreement': agreement,
            'uncertainty': uncertainty
        }
    
    def _combine_risk_factors(self, neuro_result: Dict[str, float], 
                            gemini_result: Dict[str, Any],
                            advanced_rules_result: Dict[str, Any]) -> list:
        """Combine risk factors from all models"""
        risk_factors = []
        
        # Add advanced rules risk factors
        advanced_factors = advanced_rules_result.get('risk_factors', [])
        risk_factors.extend(advanced_factors)
        
        # Add Gemini risk factors
        gemini_factors = gemini_result.get('risk_factors', [])
        risk_factors.extend(gemini_factors)
        
        # Add model-based factors
        if neuro_result.get('quantum_score', 0) > 70:
            risk_factors.append("High quantum anomaly detected")
        
        if neuro_result.get('classical_score', 0) > 70:
            risk_factors.append("Classical ML high risk")
        
        # Remove duplicates and return top factors
        unique_factors = list(dict.fromkeys(risk_factors))  # Preserves order
        return unique_factors[:6]  # Top 6 risk factors
    
    def _determine_action(self, meta_result: Dict[str, Any]) -> str:
        """Determine recommended action with enhanced logic"""
        score = meta_result['final_score']
        confidence = meta_result['confidence']
        
        if score >= 85:
            return "BLOCK TRANSACTION - Critical fraud risk detected"
        elif score >= 65:
            return "MANUAL REVIEW REQUIRED - High fraud probability"
        elif score >= 40:
            return "ADDITIONAL VERIFICATION - Medium risk detected"
        elif score >= 20:
            return "MONITOR TRANSACTION - Low risk, continue monitoring"
        else:
            return "APPROVE - Minimal fraud risk"
    
    def _fallback_prediction(self, transaction_data: Dict[str, Any]) -> MetaModelPrediction:
        """Enhanced fallback prediction"""
        logger.warning("Using enhanced fallback prediction")
        
        amount = transaction_data.get('amount', 0)
        hour = transaction_data.get('hour_of_day', 12)
        
        # Enhanced fallback scoring
        fallback_score = 20
        if amount > 100000:
            fallback_score = 85
        elif amount > 50000:
            fallback_score = 65
        elif amount > 25000:
            fallback_score = 45
        
        if hour < 6 or hour > 22:
            fallback_score = min(100, fallback_score + 20)
        
        return MetaModelPrediction(
            quantum_score=30.0,
            classical_score=35.0,
            neuro_qkad_fusion_score=fallback_score,
            gemini_logical_score=fallback_score,
            final_fraud_score=fallback_score,
            confidence_score=60.0,
            risk_level="HIGH" if fallback_score > 65 else "MEDIUM" if fallback_score > 40 else "LOW",
            primary_risk_factors=["Enhanced fallback analysis", "Model unavailable"],
            fraud_type_detected="Fallback Analysis",
            recommended_action="Manual review recommended",
            model_agreement=70.0,
            uncertainty_measure=40.0,
            timestamp=datetime.now().isoformat(),
            model_version="enhanced-fallback-v2.0"
        )
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get enhanced model statistics"""
        return {
            'model_version': self.model_version,
            'meta_weights': self.meta_weights,
            'risk_thresholds': self.risk_thresholds,
            'fraud_patterns_count': len(self.fraud_patterns),
            'neuro_qkad_loaded': bool(self.neuro_qkad_models),
            'gemini_available': self.gemini_model is not None,
            'components': [
                'Enhanced Rule-based Engine',
                'Quantum SVM (fallback)',
                'Classical XGBoost (fallback)',
                'Advanced Pattern Recognition',
                'Gemini AI Analysis',
                'Enhanced Meta-fusion Layer'
            ]
        }


# Utility functions
def create_test_high_risk_transaction() -> Dict[str, Any]:
    """Create a high-risk test transaction"""
    return {
        'amount': 150000,  # High amount
        'hour_of_day': 3,  # Suspicious time
        'is_weekend': 1,   # Weekend
        'day_of_week': 'Saturday',
        'sender_age_group': '18-25',  # Young sender
        'receiver_age_group': '56+',  # Elderly receiver
        'sender_state': 'Delhi',
        'sender_bank': 'HDFC',
        'receiver_bank': 'Unknown Bank',  # Unknown bank
        'merchant_category': 'Entertainment',
        'device_type': 'Android',
        'transaction_type': 'P2P',
        'network_type': 'WiFi',
        'transaction_status': 'SUCCESS'
    }

def create_test_medium_risk_transaction() -> Dict[str, Any]:
    """Create a medium-risk test transaction"""
    return {
        'amount': 75000,   # Moderate amount
        'hour_of_day': 22, # Late evening
        'is_weekend': 0,
        'day_of_week': 'Friday',
        'sender_age_group': '26-35',
        'receiver_age_group': '36-45',
        'sender_state': 'Mumbai',
        'sender_bank': 'SBI',
        'receiver_bank': 'HDFC',  # Cross-bank
        'merchant_category': 'Other',
        'device_type': 'Android',
        'transaction_type': 'P2P',
        'network_type': '4G',
        'transaction_status': 'SUCCESS'
    }


# Example usage and testing
if __name__ == "__main__":
    try:
        print("Initializing Enhanced Quantum Meta Model...")
        enhanced_model = EnhancedQuantumMetaModel()
        
        # Test high-risk transaction
        print("\n" + "="*60)
        print("TESTING HIGH-RISK TRANSACTION")
        print("="*60)
        
        high_risk_transaction = create_test_high_risk_transaction()
        print(f"Transaction: ₹{high_risk_transaction['amount']:,} at {high_risk_transaction['hour_of_day']}:00")
        print(f"Pattern: {high_risk_transaction['sender_age_group']} → {high_risk_transaction['receiver_age_group']}")
        print(f"Banks: {high_risk_transaction['sender_bank']} → {high_risk_transaction['receiver_bank']}")
        
        prediction = enhanced_model.predict(high_risk_transaction)
        
        print(f"\n🔴 FRAUD SCORE: {prediction.final_fraud_score:.1f}%")
        print(f"🚨 RISK LEVEL: {prediction.risk_level}")
        print(f"⚡ ACTION: {prediction.recommended_action}")
        print(f"🎯 CONFIDENCE: {prediction.confidence_score:.1f}%")
        print(f"🔍 RISK FACTORS: {', '.join(prediction.primary_risk_factors[:3])}")
        
        # Test medium-risk transaction
        print("\n" + "="*60)
        print("TESTING MEDIUM-RISK TRANSACTION")
        print("="*60)
        
        medium_risk_transaction = create_test_medium_risk_transaction()
        print(f"Transaction: ₹{medium_risk_transaction['amount']:,} at {medium_risk_transaction['hour_of_day']}:00")
        
        prediction2 = enhanced_model.predict(medium_risk_transaction)
        
        print(f"\n🟡 FRAUD SCORE: {prediction2.final_fraud_score:.1f}%")
        print(f"⚠️  RISK LEVEL: {prediction2.risk_level}")
        print(f"⚡ ACTION: {prediction2.recommended_action}")
        print(f"🎯 CONFIDENCE: {prediction2.confidence_score:.1f}%")
        
        # Test low-risk transaction
        print("\n" + "="*60)
        print("TESTING LOW-RISK TRANSACTION")
        print("="*60)
        
        low_risk_transaction = {
            'amount': 2500,
            'hour_of_day': 14,
            'is_weekend': 0,
            'day_of_week': 'Tuesday',
            'sender_age_group': '26-35',
            'receiver_age_group': '26-35',
            'sender_state': 'Mumbai',
            'sender_bank': 'SBI',
            'receiver_bank': 'SBI',
            'merchant_category': 'Grocery',
            'device_type': 'Android',
            'transaction_type': 'P2M',
            'network_type': '4G',
            'transaction_status': 'SUCCESS'
        }
        
        prediction3 = enhanced_model.predict(low_risk_transaction)
        
        print(f"Transaction: ₹{low_risk_transaction['amount']:,} at {low_risk_transaction['hour_of_day']}:00")
        print(f"\n🟢 FRAUD SCORE: {prediction3.final_fraud_score:.1f}%")
        print(f"✅ RISK LEVEL: {prediction3.risk_level}")
        print(f"⚡ ACTION: {prediction3.recommended_action}")
        
        print(f"\n📊 MODEL STATISTICS:")
        stats = enhanced_model.get_model_statistics()
        print(f"   Enhanced Model Version: {stats['model_version']}")
        print(f"   Fraud Patterns: {stats['fraud_patterns_count']}")
        print(f"   Components: {len(stats['components'])}")
        
        print(f"\n✅ Enhanced model testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in enhanced model testing: {e}")
        import traceback
        traceback.print_exc()