#!/usr/bin/env python3
"""
Test the fixed Gemini model
"""

from gemini_logical_model import GeminiLogicalModel, create_transaction_context

def test_fixed_gemini():
    """Test the fixed Gemini model"""
    
    print("🧪 Testing Fixed Gemini Model")
    print("=" * 40)
    
    try:
        # Initialize model
        print("🔧 Initializing Gemini model...")
        gemini_model = GeminiLogicalModel("AIzaSyCjKN-Cmn4PDcJoM_rV--5idHdD6NxP3tE")
        
        # Test transaction
        test_transaction = {
            'amount': 85000,
            'hour_of_day': 2,
            'is_weekend': 1,
            'day_of_week': 'Saturday',
            'sender_age_group': '18-25',
            'receiver_age_group': '56+',
            'sender_state': 'Delhi',
            'sender_bank': 'HDFC',
            'receiver_bank': 'Unknown Bank',
            'merchant_category': 'Entertainment',
            'device_type': 'Android',
            'transaction_type': 'P2P',
            'network_type': 'WiFi',
            'transaction_status': 'SUCCESS'
        }
        
        print("📊 Testing transaction analysis...")
        print(f"   Amount: ₹{test_transaction['amount']:,}")
        print(f"   Time: {test_transaction['hour_of_day']}:00 on {test_transaction['day_of_week']}")
        print(f"   Age: {test_transaction['sender_age_group']} → {test_transaction['receiver_age_group']}")
        
        # Create transaction context
        context = create_transaction_context(test_transaction)
        
        # Analyze transaction
        result = gemini_model.analyze_transaction(context)
        
        print(f"\n✅ GEMINI ANALYSIS RESULTS:")
        print(f"   Fraud Score: {result['fraud_score']}%")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Risk Factors: {', '.join(result['risk_factors'][:3])}")
        print(f"   Fraud Type: {result['fraud_type']}")
        print(f"   Recommended Action: {result['recommended_action']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_fixed_gemini()
    
    if success:
        print(f"\n🎉 Gemini model is now working correctly!")
    else:
        print(f"\n⚠️  Gemini model still has issues")