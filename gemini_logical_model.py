"""
Gemini AI-Powered Logical Fraud Detection Model
Uses Google's Gemini API to analyze transaction patterns and detect fraud
based on cybercrime documents and fraud case patterns.
"""
import google.generativeai as genai
import json
import numpy as np
from typing import Dict, Any, List
import logging
from datetime import datetime
import asyncio
import aiohttp
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TransactionContext:
    """Enhanced transaction context for Gemini analysis"""
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'amount': self.amount,
            'hour_of_day': self.hour_of_day,
            'is_weekend': self.is_weekend,
            'day_of_week': self.day_of_week,
            'sender_age_group': self.sender_age_group,
            'receiver_age_group': self.receiver_age_group,
            'sender_state': self.sender_state,
            'sender_bank': self.sender_bank,
            'receiver_bank': self.receiver_bank,
            'merchant_category': self.merchant_category,
            'device_type': self.device_type,
            'transaction_type': self.transaction_type,
            'network_type': self.network_type,
            'transaction_status': self.transaction_status
        }

class GeminiLogicalModel:
    """
    Advanced fraud detection using Gemini AI with cybercrime knowledge
    """
    
    def __init__(self, api_key: str):
        """Initialize Gemini model with API key"""
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Initialize Gemini model (using working model based on quota)
        try:
            # Use Gemini 1.5 Flash Latest (works with current quota)
            self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        except Exception as e:
            try:
                # Fallback to basic Gemini 1.5 Flash
                self.model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e2:
                try:
                    # Fallback to Gemini 2.0 Flash (if available)
                    self.model = genai.GenerativeModel('gemini-2.0-flash')
                except Exception as e3:
                    logger.error(f"All Gemini models failed: {e}, {e2}, {e3}")
                    raise Exception("No working Gemini model available")
        
        # Fraud detection knowledge base
        self.fraud_patterns_prompt = self._build_fraud_knowledge_base()
        
        logger.info(f"Gemini Logical Model initialized successfully with model: {self.model.model_name}")
    
    def list_available_models(self):
        """List available Gemini models"""
        try:
            models = genai.list_models()
            available_models = []
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    available_models.append(model.name)
            logger.info(f"Available Gemini models: {available_models}")
            return available_models
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def _build_fraud_knowledge_base(self) -> str:
        """Build comprehensive fraud detection knowledge base"""
        return """
        You are an expert fraud detection analyst with deep knowledge of cybercrime patterns, 
        financial fraud schemes, and UPI transaction anomalies. Your task is to analyze 
        transaction details and provide a fraud risk score based on the following expertise:

        ## CYBERCRIME PATTERNS & FRAUD INDICATORS:

        ### 1. TIMING-BASED FRAUD PATTERNS:
        - Late night transactions (11 PM - 6 AM): Higher fraud risk, especially for large amounts
        - Weekend transactions: Increased risk for entertainment/gambling categories
        - Holiday periods: Elevated fraud activity during festivals
        - Rapid successive transactions: Potential account takeover

        ### 2. AMOUNT-BASED SUSPICIOUS PATTERNS:
        - Round number amounts (10000, 50000): Often used in fraud
        - Just-below-limit amounts (49999 for 50000 limit): Limit testing
        - Micro-transactions followed by large amounts: Card testing pattern
        - Amounts matching common scam values: 1999, 2999, 4999

        ### 3. GEOGRAPHIC & DEMOGRAPHIC RISKS:
        - Cross-state transactions: Higher risk, especially to high-crime states
        - Age group mismatches: Young to elderly transfers (romance scams)
        - High-risk states: States with higher cybercrime rates
        - Rural to urban transfers: Potential mule account activity

        ### 4. BANK & PAYMENT METHOD RISKS:
        - Cross-bank transactions: Slightly elevated risk
        - New bank combinations: Unusual sender-receiver bank pairs
        - Failed transaction attempts: Multiple failed attempts indicate fraud
        - Payment app vulnerabilities: Certain apps more targeted

        ### 5. MERCHANT & CATEGORY RISKS:
        - Entertainment + Late night: Gambling/betting fraud
        - Online merchants: Higher fraud risk than physical stores
        - Fuel transactions: Often used for money laundering
        - Utility payments: Bill payment fraud schemes

        ### 6. DEVICE & NETWORK ANOMALIES:
        - iOS devices: Generally lower fraud risk than Android
        - 3G/older networks: Higher risk due to security vulnerabilities
        - WiFi transactions: Potential man-in-the-middle attacks
        - Device type mismatches: Unusual for user profile

        ### 7. EMERGING FRAUD TRENDS (2024-2025):
        - AI-generated fake identities
        - SIM swap attacks targeting UPI
        - Social engineering via messaging apps
        - Cryptocurrency conversion fraud
        - Digital arrest scams
        - Investment fraud via UPI
        - Romance scams targeting elderly
        - Job scam advance payments

        ### 8. BEHAVIORAL ANOMALY INDICATORS:
        - First-time large transactions
        - Unusual merchant categories for user
        - Time zone inconsistencies
        - Rapid account draining patterns
        - Beneficiary account age mismatches

        ## ANALYSIS FRAMEWORK:
        Analyze the given transaction and provide:
        1. Risk factors identified
        2. Fraud probability score (0-100)
        3. Confidence level (High/Medium/Low)
        4. Specific fraud type suspected (if any)
        5. Recommended action

        Be thorough but concise. Focus on actionable insights.
        """
    
    def analyze_transaction(self, transaction: TransactionContext) -> Dict[str, Any]:
        """
        Analyze transaction using Gemini AI for fraud detection
        
        Args:
            transaction: Transaction context with all features
            
        Returns:
            Dictionary with fraud analysis results
        """
        try:
            # Build analysis prompt
            analysis_prompt = self._build_analysis_prompt(transaction)
            
            # Get Gemini analysis
            response = self.model.generate_content(analysis_prompt)
            
            # Parse response
            analysis_result = self._parse_gemini_response(response.text)
            
            # Add metadata
            analysis_result['timestamp'] = datetime.now().isoformat()
            analysis_result['model_version'] = 'gemini-logical-v1.0'
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in Gemini analysis: {e}")
            return self._fallback_analysis(transaction)
    
    def _build_analysis_prompt(self, transaction: TransactionContext) -> str:
        """Build detailed analysis prompt for Gemini"""
        
        transaction_summary = f"""
        TRANSACTION TO ANALYZE:
        
        Amount: ₹{transaction.amount:,.2f}
        Time: {transaction.hour_of_day}:00 on {transaction.day_of_week}
        Weekend: {'Yes' if transaction.is_weekend else 'No'}
        
        Sender Profile:
        - Age Group: {transaction.sender_age_group}
        - State: {transaction.sender_state}
        - Bank: {transaction.sender_bank}
        
        Receiver Profile:
        - Age Group: {transaction.receiver_age_group}
        - Bank: {transaction.receiver_bank}
        
        Transaction Details:
        - Type: {transaction.transaction_type}
        - Category: {transaction.merchant_category}
        - Status: {transaction.transaction_status}
        - Device: {transaction.device_type}
        - Network: {transaction.network_type}
        """
        
        analysis_request = """
        Based on your cybercrime expertise and the fraud patterns above, analyze this transaction.
        
        Provide your response in the following JSON format:
        {
            "fraud_score": <0-100 integer>,
            "confidence": "<High/Medium/Low>",
            "risk_factors": ["factor1", "factor2", ...],
            "fraud_type": "<suspected fraud type or 'None'>",
            "reasoning": "<detailed explanation>",
            "recommended_action": "<action to take>",
            "severity": "<Low/Medium/High/Critical>"
        }
        
        Focus on:
        1. Timing anomalies and patterns
        2. Amount-based suspicious indicators
        3. Geographic and demographic risks
        4. Emerging fraud trends
        5. Behavioral anomalies
        
        Be precise and actionable in your analysis.
        """
        
        return self.fraud_patterns_prompt + "\n\n" + transaction_summary + "\n\n" + analysis_request
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini response and extract structured data"""
        try:
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                parsed_response = json.loads(json_str)
                
                # Validate and normalize response
                return self._validate_response(parsed_response)
            else:
                # Fallback parsing if JSON not found
                return self._extract_fallback_response(response_text)
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            return self._extract_fallback_response(response_text)
    
    def _validate_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize Gemini response"""
        validated = {
            'fraud_score': max(0, min(100, int(response.get('fraud_score', 0)))),
            'confidence': response.get('confidence', 'Medium'),
            'risk_factors': response.get('risk_factors', []),
            'fraud_type': response.get('fraud_type', 'Unknown'),
            'reasoning': response.get('reasoning', 'No detailed reasoning provided'),
            'recommended_action': response.get('recommended_action', 'Monitor transaction'),
            'severity': response.get('severity', 'Medium')
        }
        
        # Ensure confidence is valid
        if validated['confidence'] not in ['High', 'Medium', 'Low']:
            validated['confidence'] = 'Medium'
            
        # Ensure severity is valid
        if validated['severity'] not in ['Low', 'Medium', 'High', 'Critical']:
            validated['severity'] = 'Medium'
        
        return validated
    
    def _extract_fallback_response(self, response_text: str) -> Dict[str, Any]:
        """Extract fraud score from unstructured response"""
        fraud_score = 50  # Default medium risk
        confidence = 'Low'
        
        # Try to extract score from text
        import re
        score_patterns = [
            r'fraud[_\s]*score[:\s]*(\d+)',
            r'score[:\s]*(\d+)',
            r'risk[:\s]*(\d+)',
            r'(\d+)%?\s*fraud',
            r'(\d+)%?\s*risk'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, response_text.lower())
            if match:
                fraud_score = max(0, min(100, int(match.group(1))))
                break
        
        # Extract risk factors
        risk_factors = []
        risk_keywords = ['suspicious', 'anomaly', 'unusual', 'high risk', 'fraud', 'scam']
        for keyword in risk_keywords:
            if keyword in response_text.lower():
                risk_factors.append(keyword)
        
        return {
            'fraud_score': fraud_score,
            'confidence': confidence,
            'risk_factors': risk_factors,
            'fraud_type': 'Analysis incomplete',
            'reasoning': response_text[:500] + '...' if len(response_text) > 500 else response_text,
            'recommended_action': 'Manual review required',
            'severity': 'Medium'
        }
    
    def _fallback_analysis(self, transaction: TransactionContext) -> Dict[str, Any]:
        """Fallback analysis when Gemini API fails"""
        logger.warning("Using fallback analysis due to Gemini API error")
        
        # Simple rule-based fallback
        risk_score = 0
        risk_factors = []
        
        # High amount risk
        if transaction.amount > 50000:
            risk_score += 30
            risk_factors.append("High amount transaction")
        
        # Late night risk
        if transaction.hour_of_day < 6 or transaction.hour_of_day > 22:
            risk_score += 20
            risk_factors.append("Late night transaction")
        
        # Weekend risk
        if transaction.is_weekend:
            risk_score += 10
            risk_factors.append("Weekend transaction")
        
        # Cross-bank risk
        if transaction.sender_bank != transaction.receiver_bank:
            risk_score += 15
            risk_factors.append("Cross-bank transaction")
        
        # Failed transaction risk
        if transaction.transaction_status == 'FAILED':
            risk_score += 25
            risk_factors.append("Failed transaction")
        
        risk_score = min(100, risk_score)
        
        return {
            'fraud_score': risk_score,
            'confidence': 'Low',
            'risk_factors': risk_factors,
            'fraud_type': 'Rule-based analysis',
            'reasoning': 'Fallback analysis used due to API unavailability',
            'recommended_action': 'Manual review recommended',
            'severity': 'Medium' if risk_score > 50 else 'Low'
        }
    
    async def analyze_transaction_async(self, transaction: TransactionContext) -> Dict[str, Any]:
        """Async version of transaction analysis"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze_transaction, transaction)
    
    def batch_analyze(self, transactions: List[TransactionContext]) -> List[Dict[str, Any]]:
        """Analyze multiple transactions in batch"""
        results = []
        for transaction in transactions:
            try:
                result = self.analyze_transaction(transaction)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing transaction: {e}")
                results.append(self._fallback_analysis(transaction))
        
        return results
    
    def get_fraud_trends(self) -> Dict[str, Any]:
        """Get current fraud trends from Gemini"""
        trends_prompt = """
        Based on your knowledge of current cybercrime and fraud trends in 2024-2025, 
        provide the top 10 emerging fraud patterns in UPI/digital payments.
        
        Format as JSON:
        {
            "trends": [
                {
                    "name": "trend name",
                    "description": "brief description",
                    "risk_level": "High/Medium/Low",
                    "indicators": ["indicator1", "indicator2"]
                }
            ],
            "summary": "overall trend summary"
        }
        """
        
        try:
            response = self.model.generate_content(trends_prompt)
            return self._parse_gemini_response(response.text)
        except Exception as e:
            logger.error(f"Error getting fraud trends: {e}")
            return {"trends": [], "summary": "Unable to fetch current trends"}


# Utility functions for integration
def create_transaction_context(transaction_data: Dict[str, Any]) -> TransactionContext:
    """Create TransactionContext from dictionary"""
    return TransactionContext(
        amount=float(transaction_data.get('amount', 0)),
        hour_of_day=int(transaction_data.get('hour_of_day', 12)),
        is_weekend=int(transaction_data.get('is_weekend', 0)),
        day_of_week=str(transaction_data.get('day_of_week', 'Monday')),
        sender_age_group=str(transaction_data.get('sender_age_group', '26-35')),
        receiver_age_group=str(transaction_data.get('receiver_age_group', '26-35')),
        sender_state=str(transaction_data.get('sender_state', 'Delhi')),
        sender_bank=str(transaction_data.get('sender_bank', 'SBI')),
        receiver_bank=str(transaction_data.get('receiver_bank', 'SBI')),
        merchant_category=str(transaction_data.get('merchant_category', 'Other')),
        device_type=str(transaction_data.get('device_type', 'Android')),
        transaction_type=str(transaction_data.get('transaction_type', 'P2P')),
        network_type=str(transaction_data.get('network_type', '4G')),
        transaction_status=str(transaction_data.get('transaction_status', 'SUCCESS'))
    )


# Example usage and testing
if __name__ == "__main__":
    # Initialize model
    API_KEY = "AIzaSyCjKN-Cmn4PDcJoM_rV--5idHdD6NxP3tE"
    gemini_model = GeminiLogicalModel(API_KEY)
    
    # Test transaction
    test_transaction = TransactionContext(
        amount=75000,
        hour_of_day=2,
        is_weekend=1,
        day_of_week="Saturday",
        sender_age_group="18-25",
        receiver_age_group="46-55",
        sender_state="Delhi",
        sender_bank="HDFC",
        receiver_bank="SBI",
        merchant_category="Entertainment",
        device_type="Android",
        transaction_type="P2P",
        network_type="WiFi",
        transaction_status="SUCCESS"
    )
    
    # Analyze transaction
    print("Analyzing suspicious transaction...")
    result = gemini_model.analyze_transaction(test_transaction)
    
    print(f"\nFraud Analysis Results:")
    print(f"Fraud Score: {result['fraud_score']}/100")
    print(f"Confidence: {result['confidence']}")
    print(f"Risk Factors: {', '.join(result['risk_factors'])}")
    print(f"Suspected Fraud Type: {result['fraud_type']}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"Recommended Action: {result['recommended_action']}")
    print(f"Severity: {result['severity']}")