#!/usr/bin/env python3
"""
Startup script for Quantum Meta Fraud Detection API
"""

import uvicorn
import sys
import os

def main():
    """Start the Quantum Fraud API server"""
    
    print("🚀 Starting Quantum Meta Fraud Detection API...")
    print("=" * 60)
    
    # Check if required files exist
    required_files = [
        "quantum_meta_model.py",
        "gemini_logical_model.py",
        "quantum_fraud_api.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return
    
    print("✅ All required files found")
    print("🔧 Initializing server...")
    
    try:
        # Start the server
        uvicorn.run(
            "quantum_fraud_api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()