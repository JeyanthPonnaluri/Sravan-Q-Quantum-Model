#!/usr/bin/env python3
"""
Enhanced Web Application Runner
Runs the quantum fraud detection web app with the enhanced engine
"""

import os
import sys
import webbrowser
import time
import threading
import uvicorn
from quantum_web_app import app

def print_startup_info():
    """Print startup information"""
    print("🚀 Starting Enhanced Quantum Fraud Detection Web Application")
    print("=" * 70)
    print("🔮 Features:")
    print("   • Enhanced Quantum Meta Model v2.0")
    print("   • Fixed Gemini AI Integration (1.5-Flash-Latest)")
    print("   • 12 Advanced Fraud Patterns")
    print("   • Real-time Risk Assessment")
    print("   • Professional Web Interface")
    print("   • Mobile Responsive Design")
    print()
    print("🎯 Fraud Detection Capabilities:")
    print("   • Elder Fraud Detection")
    print("   • Velocity Fraud (High Amount)")
    print("   • Time-based Fraud (Late Night)")
    print("   • Cross-bank Fraud")
    print("   • Pattern Combination Analysis")
    print()
    print("📊 Risk Levels:")
    print("   • MINIMAL (0-25%): Approve")
    print("   • LOW (25-45%): Monitor")
    print("   • MEDIUM (45-70%): Additional Verification")
    print("   • HIGH (70-85%): Manual Review")
    print("   • CRITICAL (85-100%): Block Transaction")
    print()

def open_browser():
    """Open browser after server starts"""
    time.sleep(3)  # Wait for server to fully start
    try:
        webbrowser.open("http://localhost:8000")
        print("🌐 Browser opened automatically")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("   Please manually visit: http://localhost:8000")

def main():
    """Main function to run the enhanced web application"""
    
    print_startup_info()
    
    # Check if required files exist
    required_files = [
        "quantum_web_app.py",
        "enhanced_quantum_model.py",
        "gemini_logical_model.py",
        "templates/index.html",
        "static/style.css"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return
    
    print("✅ All required files found")
    print("🔧 Starting enhanced web server...")
    print("📱 Web interface will be available at: http://localhost:8000")
    print("🌐 Opening browser automatically in 3 seconds...")
    print()
    print("💡 Test Examples:")
    print("   • High Risk: ₹150,000 at 3:00 AM, Young→Elder")
    print("   • Medium Risk: ₹75,000 at 11:00 PM, Weekend")
    print("   • Low Risk: ₹2,500 at 2:00 PM, Same Bank")
    print()
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 70)
    
    # Open browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        # Start the enhanced web application
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Enhanced web application stopped by user")
        print("✅ Thank you for using Quantum Fraud Detection!")
    except Exception as e:
        print(f"❌ Error starting enhanced web application: {e}")

if __name__ == "__main__":
    main()