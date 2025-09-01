#!/usr/bin/env python3
"""
Demo script for Quantum Meta Fraud Detection API
Shows various usage examples and scenarios
"""

import requests
import json
import time
from typing import Dict, Any

# API Configuration
API_BASE_URL = "http://localhost:8000"

def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"🔮 {title}")
    print(f"{'='*60}")

def print_result(transaction: Dict[str, Any], result: Dict[str, Any]):
    """Print formatted analysis result"""
    print(f"\n💰 Transaction: ₹{transaction['amount']:,} at {transaction.get('hour_of_day', 'N/A')}:00")
    print(f"📊 Fraud Score: {result['fraud_score']}%")
    print(f"⚠️  Risk Level: {result['risk_level']}")
    print(f"🎯 Action: {result['recommended_action']}")
    
    if 'individual_scores' in result:
        print(f"\n📈 Model Breakdown:")
        for model, score in result['individual_scores'].items():
            model_name = model.replace('_', ' ').title()
            print(f"   {model_name}: {score}%")
    
    if 'primary_risk_factors' in result and result['primary_risk_factors']:
        print(f"\n🚨 Risk Factors: {', '.join(result['primary_risk_factors'][:3])}")

def demo_low_risk_transaction():
    """Demo: Low risk transaction"""
    print_header("LOW RISK TRANSACTION DEMO")
    
    transaction = {
        "amount": 2500,
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
    
    try:
        response = requests.post(f"{API_BASE_URL}/analyze", json=transaction)
        if response.status_code == 200:
            result = response.json()
            print_result(transaction, result)
            print(f"⏱️  Processing Time: {result['processing_time_ms']}ms")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {e}")

def demo_high_risk_transaction():
    """Demo: High risk transaction"""
    print_header("HIGH RISK TRANSACTION DEMO")
    
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
    
    try:
        response = requests.post(f"{API_BASE_URL}/analyze", json=transaction)
        if response.status_code == 200:
            result = response.json()
            print_result(transaction, result)
            print(f"⏱️  Processing Time: {result['processing_time_ms']}ms")
            print(f"🔍 Confidence: {result['confidence_score']}%")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {e}")

def demo_quick_scoring():
    """Demo: Quick scoring for multiple transactions"""
    print_header("QUICK SCORING DEMO")
    
    transactions = [
        {"amount": 1000, "hour_of_day": 10, "description": "Small daytime transaction"},
        {"amount": 50000, "hour_of_day": 15, "description": "Large afternoon transaction"},
        {"amount": 75000, "hour_of_day": 2, "description": "Large late-night transaction"},
        {"amount": 5000, "hour_of_day": 22, "description": "Medium evening transaction"}
    ]
    
    print("Analyzing multiple transactions for quick scoring...")
    
    for i, trans in enumerate(transactions, 1):
        description = trans.pop('description')
        
        try:
            start_time = time.time()
            response = requests.post(f"{API_BASE_URL}/score", json=trans)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                processing_time = (end_time - start_time) * 1000
                
                print(f"\n{i}. {description}")
                print(f"   Amount: ₹{trans['amount']:,} | Time: {trans['hour_of_day']}:00")
                print(f"   Score: {result['fraud_score']}% | Risk: {result['risk_level']}")
                print(f"   Response Time: {processing_time:.1f}ms")
            else:
                print(f"   ❌ Error: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Request failed: {e}")

def demo_batch_analysis():
    """Demo: Batch analysis"""
    print_header("BATCH ANALYSIS DEMO")
    
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
        },
        {
            "amount": 45000,
            "hour_of_day": 23,
            "is_weekend": 0,
            "day_of_week": "Thursday",
            "sender_age_group": "26-35",
            "receiver_age_group": "18-25"
        }
    ]
    
    batch_request = {"transactions": transactions}
    
    try:
        print(f"Analyzing batch of {len(transactions)} transactions...")
        response = requests.post(f"{API_BASE_URL}/analyze/batch", json=batch_request)
        
        if response.status_code == 200:
            result = response.json()
            summary = result['summary']
            
            print(f"\n📊 BATCH SUMMARY:")
            print(f"   Total Transactions: {summary['total_transactions']}")
            print(f"   Successful Analyses: {summary['successful_analyses']}")
            print(f"   Average Fraud Score: {summary['average_fraud_score']}%")
            print(f"   High Risk Count: {summary['high_risk_count']}")
            print(f"   Total Processing Time: {summary['processing_time_ms']}ms")
            
            print(f"\n📋 INDIVIDUAL RESULTS:")
            for i, (trans, res) in enumerate(zip(transactions, result['results']), 1):
                risk_emoji = "🟢" if res['risk_level'] in ["MINIMAL", "LOW"] else "🟡" if res['risk_level'] == "MEDIUM" else "🔴"
                print(f"   {i}. ₹{trans['amount']:,} at {trans['hour_of_day']}:00 → {res['fraud_score']}% {risk_emoji} ({res['risk_level']})")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {e}")

def demo_api_info():
    """Demo: API information and health"""
    print_header("API INFORMATION")
    
    try:
        # Health check
        print("🏥 Health Check:")
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"   Status: {health['status']}")
            print(f"   Model Loaded: {health['model_loaded']}")
            print(f"   Version: {health['version']}")
        
        # Model stats
        print("\n📊 Model Statistics:")
        response = requests.get(f"{API_BASE_URL}/model/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"   Model Version: {stats['model_version']}")
            print(f"   Components: {len(stats['components'])}")
            print(f"   Neuro-QKAD Ready: {stats['neuro_qkad_loaded']}")
            print(f"   Gemini Available: {stats['gemini_available']}")
            
            print(f"\n⚖️  Model Weights:")
            for component, weight in stats['meta_weights'].items():
                print(f"   {component.replace('_', ' ').title()}: {weight}")
        
    except Exception as e:
        print(f"❌ Request failed: {e}")

def demo_performance_test():
    """Demo: Performance testing"""
    print_header("PERFORMANCE TEST")
    
    transaction = {
        "amount": 25000,
        "hour_of_day": 12,
        "is_weekend": 0,
        "day_of_week": "Thursday"
    }
    
    num_requests = 5
    times = []
    
    print(f"Running {num_requests} requests to test performance...")
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(f"{API_BASE_URL}/score", json=transaction)
            end_time = time.time()
            
            if response.status_code == 200:
                response_time = (end_time - start_time) * 1000
                times.append(response_time)
                result = response.json()
                print(f"   Request {i+1}: {response_time:.1f}ms → Score: {result['fraud_score']}%")
            else:
                print(f"   Request {i+1}: Error {response.status_code}")
        except Exception as e:
            print(f"   Request {i+1}: Failed - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n⏱️  PERFORMANCE SUMMARY:")
        print(f"   Average Response Time: {avg_time:.1f}ms")
        print(f"   Fastest Response: {min_time:.1f}ms")
        print(f"   Slowest Response: {max_time:.1f}ms")
        print(f"   Success Rate: {len(times)}/{num_requests} ({len(times)/num_requests*100:.1f}%)")

def main():
    """Run all demos"""
    print("🚀 Quantum Meta Fraud Detection API Demo")
    print("Make sure the API server is running on http://localhost:8000")
    
    # Check if API is available
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API server is not responding correctly")
            print("Start the server with: python quantum_fraud_api.py")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("Start the server with: python quantum_fraud_api.py")
        return
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        return
    
    print("✅ API server is running")
    
    # Run all demos
    demo_api_info()
    demo_low_risk_transaction()
    demo_high_risk_transaction()
    demo_quick_scoring()
    demo_batch_analysis()
    demo_performance_test()
    
    print_header("DEMO COMPLETED")
    print("🎉 All demos completed successfully!")
    print("\nNext steps:")
    print("1. Visit http://localhost:8000/docs for interactive API documentation")
    print("2. Try your own transactions using the /analyze endpoint")
    print("3. Integrate the API into your fraud detection system")
    print("\nFor more information, see QUANTUM_API_DOCS.md")

if __name__ == "__main__":
    main()