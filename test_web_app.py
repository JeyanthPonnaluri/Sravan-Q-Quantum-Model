#!/usr/bin/env python3
"""
Test script for Quantum Fraud Detection Web Application
"""

import time
import threading
import requests
import uvicorn
from quantum_web_app import app

def test_web_endpoints():
    """Test web application endpoints"""
    base_url = "http://localhost:8000"
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(3)
    
    try:
        # Test home page
        print("Testing home page...")
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            print("✅ Home page loads successfully")
            if "Quantum Fraud Detection" in response.text:
                print("✅ Home page contains expected content")
            else:
                print("⚠️ Home page missing expected content")
        else:
            print(f"❌ Home page failed: {response.status_code}")
            return False
        
        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed - Status: {health_data['status']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
        
        # Test form submission
        print("Testing form submission...")
        form_data = {
            "amount": 25000,
            "hour_of_day": 14,
            "is_weekend": 0,
            "day_of_week": "Tuesday",
            "sender_age_group": "26-35",
            "receiver_age_group": "26-35",
            "sender_state": "Delhi",
            "sender_bank": "SBI",
            "receiver_bank": "SBI",
            "merchant_category": "Other",
            "device_type": "Android",
            "transaction_type": "P2P",
            "network_type": "4G",
            "transaction_status": "SUCCESS"
        }
        
        response = requests.post(f"{base_url}/analyze", data=form_data, timeout=15)
        if response.status_code == 200:
            print("✅ Form submission successful")
            if "Fraud Analysis Results" in response.text:
                print("✅ Results page displays correctly")
            else:
                print("⚠️ Results page missing expected content")
        else:
            print(f"❌ Form submission failed: {response.status_code}")
            if response.text:
                print(f"Error details: {response.text[:200]}...")
        
        return True
        
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
    print("🧪 Quantum Fraud Detection Web App Test")
    print("=" * 50)
    
    # Start server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Run tests
    success = test_web_endpoints()
    
    if success:
        print("\n🎉 Web application is working correctly!")
        print("\nTo start the full web application, run:")
        print("   python run_web_app.py")
        print("\nOr directly:")
        print("   python quantum_web_app.py")
        print("\nThen visit: http://localhost:8000")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
    
    print("\nTest completed.")