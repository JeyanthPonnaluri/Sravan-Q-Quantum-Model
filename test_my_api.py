#!/usr/bin/env python3
"""
Simple test script for the Quantum Fraud API
"""

import requests
import json

# API base URL
API_URL = "http://localhost:8000"

def test_fraud_detection():
    """Test fraud detection with different scenarios"""
    
    print("🔍 Testing Quantum Meta Fraud Detection API")
    print("=" * 50)
    
    # Test 1: Low Risk Transaction
    print("\n1️⃣ Testing LOW RISK transaction:")
    low_risk = {
        "amount": 2500,
        "hour_of_day": 14,
        "is_weekend": 0,
        "day_of_week": "Tuesday",
        "sender_age_group": "26-35",
        "receiver_age_group": "26-35"
    }
    
    response = requests.post(f"{API_URL}/score", json=low_risk)
    if response.status_code == 200:
        result = response.json()
        print(f"   💰 Amount: ₹{low_risk['amount']:,}")
        print(f"   📊 Fraud Score: {result['fraud_score']}%")
        print(f"   ⚠️  Risk Level: {result['risk_level']}")
    else:
        print(f"   ❌ Error: {response.status_code}")
    
    # Test 2: High Risk Transaction
    print("\n2️⃣ Testing HIGH RISK transaction:")
    high_risk = {
        "amount": 95000,
        "hour_of_day": 3,
        "is_weekend": 1,
        "day_of_week": "Saturday",
        "sender_age_group": "18-25",
        "receiver_age_group": "56+"
    }
    
    response = requests.post(f"{API_URL}/analyze", json=high_risk)
    if response.status_code == 200:
        result = response.json()
        print(f"   💰 Amount: ₹{high_risk['amount']:,}")
        print(f"   📊 Fraud Score: {result['fraud_score']}%")
        print(f"   ⚠️  Risk Level: {result['risk_level']}")
        print(f"   🎯 Action: {result['recommended_action']}")
        print(f"   🚨 Risk Factors: {', '.join(result['primary_risk_factors'][:3])}")
        print(f"   ⏱️  Processing Time: {result['processing_time_ms']}ms")
    else:
        print(f"   ❌ Error: {response.status_code}")
    
    # Test 3: Batch Analysis
    print("\n3️⃣ Testing BATCH analysis:")
    batch_transactions = {
        "transactions": [
            {"amount": 5000, "hour_of_day": 10, "is_weekend": 0},
            {"amount": 75000, "hour_of_day": 2, "is_weekend": 1},
            {"amount": 2500, "hour_of_day": 15, "is_weekend": 0}
        ]
    }
    
    response = requests.post(f"{API_URL}/analyze/batch", json=batch_transactions)
    if response.status_code == 200:
        result = response.json()
        summary = result['summary']
        print(f"   📦 Total Transactions: {summary['total_transactions']}")
        print(f"   ✅ Successful: {summary['successful_analyses']}")
        print(f"   📊 Average Score: {summary['average_fraud_score']}%")
        print(f"   🔴 High Risk Count: {summary['high_risk_count']}")
        
        print(f"\n   📋 Individual Results:")
        for i, res in enumerate(result['results'], 1):
            print(f"      {i}. Score: {res['fraud_score']}% ({res['risk_level']})")
    else:
        print(f"   ❌ Error: {response.status_code}")
    
    print(f"\n✅ Testing completed!")

if __name__ == "__main__":
    test_fraud_detection()