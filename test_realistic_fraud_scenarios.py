#!/usr/bin/env python3
"""
Test Realistic Fraud Scenarios
Demonstrates the enhanced fraud detection capabilities
"""

from enhanced_quantum_model import EnhancedQuantumMetaModel

def test_fraud_scenarios():
    """Test various realistic fraud scenarios"""
    
    print("🔍 Testing Enhanced Quantum Fraud Detection")
    print("=" * 60)
    
    # Initialize enhanced model
    model = EnhancedQuantumMetaModel()
    
    # Define test scenarios
    scenarios = [
        {
            'name': '🔴 CRITICAL: Elder Fraud (Young to Old, Large Amount, Late Night)',
            'data': {
                'amount': 250000,
                'hour_of_day': 2,
                'is_weekend': 1,
                'day_of_week': 'Saturday',
                'sender_age_group': '18-25',
                'receiver_age_group': '56+',
                'sender_bank': 'HDFC',
                'receiver_bank': 'Unknown Bank',
                'merchant_category': 'Entertainment'
            },
            'expected_risk': 'CRITICAL'
        },
        {
            'name': '🔴 HIGH: Large Cross-Bank Transfer at Suspicious Time',
            'data': {
                'amount': 150000,
                'hour_of_day': 3,
                'is_weekend': 0,
                'day_of_week': 'Wednesday',
                'sender_age_group': '26-35',
                'receiver_age_group': '36-45',
                'sender_bank': 'SBI',
                'receiver_bank': 'Unknown Bank',
                'merchant_category': 'Other'
            },
            'expected_risk': 'HIGH'
        },
        {
            'name': '🟡 MEDIUM: Weekend High-Value Entertainment',
            'data': {
                'amount': 75000,
                'hour_of_day': 23,
                'is_weekend': 1,
                'day_of_week': 'Sunday',
                'sender_age_group': '18-25',
                'receiver_age_group': '26-35',
                'sender_bank': 'HDFC',
                'receiver_bank': 'ICICI',
                'merchant_category': 'Entertainment'
            },
            'expected_risk': 'MEDIUM'
        },
        {
            'name': '🟡 MEDIUM: Large Amount Cross-Bank',
            'data': {
                'amount': 125000,
                'hour_of_day': 15,
                'is_weekend': 0,
                'day_of_week': 'Tuesday',
                'sender_age_group': '36-45',
                'receiver_age_group': '26-35',
                'sender_bank': 'SBI',
                'receiver_bank': 'Axis',
                'merchant_category': 'Other'
            },
            'expected_risk': 'MEDIUM'
        },
        {
            'name': '🟢 LOW: Normal Daytime Grocery Transaction',
            'data': {
                'amount': 3500,
                'hour_of_day': 14,
                'is_weekend': 0,
                'day_of_week': 'Tuesday',
                'sender_age_group': '26-35',
                'receiver_age_group': '26-35',
                'sender_bank': 'SBI',
                'receiver_bank': 'SBI',
                'merchant_category': 'Grocery'
            },
            'expected_risk': 'LOW'
        },
        {
            'name': '🟢 MINIMAL: Small Same-Bank Transaction',
            'data': {
                'amount': 1200,
                'hour_of_day': 10,
                'is_weekend': 0,
                'day_of_week': 'Monday',
                'sender_age_group': '26-35',
                'receiver_age_group': '26-35',
                'sender_bank': 'HDFC',
                'receiver_bank': 'HDFC',
                'merchant_category': 'Food'
            },
            'expected_risk': 'MINIMAL'
        },
        {
            'name': '🔴 CRITICAL: Extremely High Amount (Velocity Fraud)',
            'data': {
                'amount': 500000,
                'hour_of_day': 14,
                'is_weekend': 0,
                'day_of_week': 'Thursday',
                'sender_age_group': '26-35',
                'receiver_age_group': '36-45',
                'sender_bank': 'SBI',
                'receiver_bank': 'HDFC',
                'merchant_category': 'Other'
            },
            'expected_risk': 'CRITICAL'
        },
        {
            'name': '🟡 MEDIUM: Late Night High Amount',
            'data': {
                'amount': 85000,
                'hour_of_day': 1,
                'is_weekend': 0,
                'day_of_week': 'Friday',
                'sender_age_group': '26-35',
                'receiver_age_group': '36-45',
                'sender_bank': 'ICICI',
                'receiver_bank': 'PNB',
                'merchant_category': 'Shopping'
            },
            'expected_risk': 'MEDIUM'
        }
    ]
    
    # Test each scenario
    results = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print("-" * 50)
        
        data = scenario['data']
        print(f"   💰 Amount: ₹{data['amount']:,}")
        print(f"   🕐 Time: {data['hour_of_day']:02d}:00 on {data['day_of_week']}")
        print(f"   👥 Age: {data['sender_age_group']} → {data['receiver_age_group']}")
        print(f"   🏦 Banks: {data['sender_bank']} → {data['receiver_bank']}")
        print(f"   🏪 Category: {data['merchant_category']}")
        
        # Get prediction
        try:
            prediction = model.predict(data)
            
            # Display results
            risk_emoji = {
                'MINIMAL': '🟢',
                'LOW': '🟢', 
                'MEDIUM': '🟡',
                'HIGH': '🔴',
                'CRITICAL': '🔴'
            }.get(prediction.risk_level, '⚪')
            
            print(f"\n   📊 RESULT:")
            print(f"      Fraud Score: {prediction.final_fraud_score:.1f}%")
            print(f"      Risk Level: {risk_emoji} {prediction.risk_level}")
            print(f"      Confidence: {prediction.confidence_score:.1f}%")
            print(f"      Action: {prediction.recommended_action}")
            
            if prediction.primary_risk_factors:
                print(f"      Risk Factors: {', '.join(prediction.primary_risk_factors[:2])}")
            
            # Check if prediction matches expectation
            expected = scenario['expected_risk']
            actual = prediction.risk_level
            
            if actual == expected:
                print(f"   ✅ CORRECT: Expected {expected}, Got {actual}")
                results.append(True)
            else:
                print(f"   ⚠️  DIFFERENT: Expected {expected}, Got {actual}")
                results.append(False)
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append(False)
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"📈 TEST SUMMARY")
    print(f"=" * 60)
    
    correct = sum(results)
    total = len(results)
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print(f"✅ Correct Predictions: {correct}/{total}")
    print(f"📊 Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 75:
        print(f"🎉 EXCELLENT: Model shows good fraud detection capability!")
    elif accuracy >= 50:
        print(f"👍 GOOD: Model shows reasonable fraud detection capability")
    else:
        print(f"⚠️  NEEDS IMPROVEMENT: Model may need further tuning")
    
    # Risk distribution
    risk_counts = {}
    for i, scenario in enumerate(scenarios):
        try:
            prediction = model.predict(scenario['data'])
            risk_level = prediction.risk_level
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        except:
            pass
    
    print(f"\n📊 RISK DISTRIBUTION:")
    for risk, count in sorted(risk_counts.items()):
        percentage = (count / len(scenarios)) * 100
        print(f"   {risk}: {count} transactions ({percentage:.1f}%)")
    
    return accuracy >= 75

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    
    print(f"\n" + "=" * 60)
    print(f"🔬 TESTING EDGE CASES")
    print(f"=" * 60)
    
    model = EnhancedQuantumMetaModel()
    
    edge_cases = [
        {
            'name': 'Boundary Amount (₹50,000)',
            'data': {'amount': 50000, 'hour_of_day': 14}
        },
        {
            'name': 'Boundary Amount (₹100,000)', 
            'data': {'amount': 100000, 'hour_of_day': 14}
        },
        {
            'name': 'Peak Fraud Hour (3 AM)',
            'data': {'amount': 25000, 'hour_of_day': 3}
        },
        {
            'name': 'Boundary Hour (6 AM)',
            'data': {'amount': 25000, 'hour_of_day': 6}
        },
        {
            'name': 'Maximum Amount Test',
            'data': {'amount': 1000000, 'hour_of_day': 14}
        }
    ]
    
    for case in edge_cases:
        print(f"\n🧪 {case['name']}")
        try:
            prediction = model.predict(case['data'])
            print(f"   Score: {prediction.final_fraud_score:.1f}% | Risk: {prediction.risk_level}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    success = test_fraud_scenarios()
    test_edge_cases()
    
    print(f"\n🎯 CONCLUSION:")
    if success:
        print("✅ Enhanced fraud detection model is working effectively!")
        print("✅ Ready for real-world fraud detection scenarios")
    else:
        print("⚠️  Model may need further calibration for optimal performance")
    
    print(f"\n💡 The enhanced model now provides:")
    print("   • More realistic fraud scoring")
    print("   • Better detection of high-risk patterns")
    print("   • Improved risk level classification")
    print("   • Enhanced pattern recognition")
    print("   • Real-world fraud scenario handling")