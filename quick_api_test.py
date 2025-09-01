#!/usr/bin/env python3
"""
Quick test to verify the Quantum Fraud API works
"""

import sys
import time
import subprocess
import requests
import threading
from quantum_fraud_api import app
import uvicorn

def test_api_endpoints():
    """Test basic API functionality"""
    base_url = "http://localhost:8000"
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(3)
    
    try:
        # Test health check
        print("Testing health check...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
        
        # Test example endpoint
        print("Testing example endpoint...")
        response = requests.get(f"{base_url}/example", timeout=10)
        if response.status_code == 200:
            print("✅ Example endpoint passed")
        else:
            print(f"❌ Example endpoint failed: {response.status_code}")
            return False
        
        # Test fraud analysis
        print("Testing fraud analysis...")
        transaction = {
            "amount": 50000,
            "hour_of_day": 14,
            "is_weekend": 0,
            "day_of_week": "Tuesday"
        }
        
        response = requests.post(f"{base_url}/score", json=transaction, timeout=15)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Fraud analysis passed - Score: {result['fraud_score']}%")
            return True
        else:
            print(f"❌ Fraud analysis failed: {response.status_code}")
            if response.text:
                print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - server may not be running")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def run_server():
    """Run the server in a separate thread"""
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")
    except Exception as e:
        print(f"Server error: {e}")

if __name__ == "__main__":
    print("🧪 Quick Quantum Fraud API Test")
    print("=" * 40)
    
    # Start server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Run tests
    success = test_api_endpoints()
    
    if success:
        print("\n🎉 All tests passed! API is working correctly.")
        print("\nTo start the full server, run:")
        print("   python quantum_fraud_api.py")
        print("\nThen visit: http://localhost:8000/docs")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
    
    print("\nTest completed.")