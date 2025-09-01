#!/usr/bin/env python3
"""
Test script for Quantum Meta Fraud Detection API
"""

import requests
import json
import time
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_model_stats():
    """Test model statistics endpoint"""
    print("\n📊 Testing model statistics...")
    response = requests.get(f"{BASE_URL}/model/stats")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        stats = response.json()
        print(f"Model Version: {stats['model_version']}")
        print(f"Components: {len(stats['components'])}")
        print(f"Neuro-QKAD Loaded: {stats['neuro_qkad_loaded']}")
        print(f"Gemini Available: {stats['gemini_available']}")
    return response.status_code == 200

def test_single_transaction():
    """Test single transaction analysis"""
    print("\n🔮 Testing single transaction analysis...")
    
    # High-risk transaction
    transaction = {
        "amount": 95000,
        "hour_of_day": 3,
        "is_weekend": 1,
        "day_of_week": "Saturday",
        "sender_age_group": "18-25",
        "receiver_age_group": "56+",
        "sender_state": "Delhi",
        "sender_bank": "HDFC",
        "receiver_bank": "Unknown Bank",
        "merchant_category": "Entertainment",
        "device_type": "Android",
        "transaction_type": "P2P",
        "network_type": "WiFi",
        "transaction_status": "SUCCESS"
    }
    
    response = requests.post(f"{BASE_URL}/analyze", json=transaction)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n🎯 ANALYSIS RESULTS:")
        print(f"   Fraud Score: {result['fraud_score']}%")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Confidence: {result['confidence_score']}%")
        print(f"   Recommended Action: {result['recommended_action']}")
        print(f"   Processing Time: {result['processing_time_ms']}ms")
        print(f"   Risk Factors: {', '.join(result['primary_risk_factors'][:3])}")
        
        print(f"\n📈 INDIVIDUAL SCORES:")
        for model, score in result['individual_scores'].items():
            print(f"   {model}: {score}%")
    
    return response.status_code == 200

def test_quick_score():
    """Test quick score endpoint"""
    print("\n⚡ Testing quick score endpoint...")
    
    # Low-risk transaction
    transaction = {
        "amount": 1500,
        "hour_of_day": 14,
        "is_weekend": 0,
        "day_of_week": "Tuesday",
        "sender_age_group": "26-35",
        "receiver_age_group": "26-35",
        "sender_state": "Mumbai",
        "sender_bank": "SBI",
        "receiver_bank": "SBI",
        "merchant_category": "Grocery",
        "device_type": "Android",
        "transaction_type": "P2M",
        "network_type": "4G",
        "transaction_status": "SUCCESS"
    }
    
    response = requests.post(f"{BASE_URL}/score", json=transaction)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Quick Score: {result['fraud_score']}%")
        print(f"Risk Level: {result['risk_level']}")
    
    return response.status_code == 200

def test_batch_analysis():
    """Test batch transaction analysis"""
    print("\n📦 Testing batch analysis...")
    
    transactions = [
        {
            "amount": 5000,
            "hour_of_day": 10,
            "is_weekend": 0,
            "day_of_week": "Monday",
            "sender_age_group": "26-35",
            "receiver_age_group": "26-35"
        },
        {
            "amount": 75000,
            "hour_of_day": 2,
            "is_weekend": 1,
            "day_of_week": "Sunday",
            "sender_age_group": "18-25",
            "receiver_age_group": "56+"
        },
        {
            "amount": 2500,
            "hour_of_day": 15,
            "is_weekend": 0,
            "day_of_week": "Wednesday",
            "sender_age_group": "36-45",
            "receiver_age_group": "36-45"
        }
    ]
    
    batch_request = {"transactions": transactions}
    response = requests.post(f"{BASE_URL}/analyze/batch", json=batch_request)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        summary = result['summary']
        
        print(f"\n📊 BATCH SUMMARY:")
        print(f"   Total Transactions: {summary['total_transactions']}")
        print(f"   Successful Analyses: {summary['successful_analyses']}")
        print(f"   Average Fraud Score: {summary['average_fraud_score']}%")
        print(f"   High Risk Count: {summary['high_risk_count']}")
        print(f"   Processing Time: {summary['processing_time_ms']}ms")
        
        print(f"\n📋 INDIVIDUAL RESULTS:")
        for i, res in enumerate(result['results']):
            print(f"   Transaction {i+1}: {res['fraud_score']}% ({res['risk_level']})")
    
    return response.status_code == 200

def test_example_endpoint():
    """Test example transaction endpoint"""
    print("\n📝 Testing example endpoint...")
    response = requests.get(f"{BASE_URL}/example")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        example = response.json()
        print("Example transaction received successfully")
        print(f"Amount: ₹{example['example_transaction']['amount']:,}")
    
    return response.status_code == 200

def run_performance_test():
    """Run performance test"""
    print("\n🚀 Running performance test...")
    
    transaction = {
        "amount": 25000,
        "hour_of_day": 12,
        "is_weekend": 0,
        "day_of_week": "Thursday"
    }
    
    num_requests = 10
    times = []
    
    for i in range(num_requests):
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/score", json=transaction)
        end_time = time.time()
        
        if response.status_code == 200:
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        print(f"Request {i+1}: {response.status_code} ({(end_time - start_time) * 1000:.2f}ms)")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n⏱️  PERFORMANCE RESULTS:")
        print(f"   Average Response Time: {avg_time:.2f}ms")
        print(f"   Min Response Time: {min_time:.2f}ms")
        print(f"   Max Response Time: {max_time:.2f}ms")
        print(f"   Successful Requests: {len(times)}/{num_requests}")

def main():
    """Run all tests"""
    print("🧪 Quantum Meta Fraud Detection API Test Suite")
    print("=" * 60)
    
    try:
        # Test basic endpoints
        health_ok = test_health_check()
        if not health_ok:
            print("❌ Health check failed - API may not be running")
            return
        
        stats_ok = test_model_stats()
        example_ok = test_example_endpoint()
        
        # Test analysis endpoints
        single_ok = test_single_transaction()
        quick_ok = test_quick_score()
        batch_ok = test_batch_analysis()
        
        # Performance test
        run_performance_test()
        
        # Summary
        print(f"\n✅ TEST SUMMARY:")
        print(f"   Health Check: {'✅' if health_ok else '❌'}")
        print(f"   Model Stats: {'✅' if stats_ok else '❌'}")
        print(f"   Example Endpoint: {'✅' if example_ok else '❌'}")
        print(f"   Single Analysis: {'✅' if single_ok else '❌'}")
        print(f"   Quick Score: {'✅' if quick_ok else '❌'}")
        print(f"   Batch Analysis: {'✅' if batch_ok else '❌'}")
        
        all_passed = all([health_ok, stats_ok, example_ok, single_ok, quick_ok, batch_ok])
        print(f"\n🎯 Overall Result: {'All tests passed! 🎉' if all_passed else 'Some tests failed ⚠️'}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the API server is running on http://localhost:8000")
        print("   Start the server with: python quantum_fraud_api.py")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()