#!/usr/bin/env python3
"""
Launch script for Quantum Fraud Detection Web Application
"""

import os
import sys
import webbrowser
import time
import threading
import uvicorn
from quantum_web_app import app

def open_browser():
    """Open browser after a short delay"""
    time.sleep(2)  # Wait for server to start
    webbrowser.open("http://localhost:8000")

def main():
    """Main function to run the web application"""
    print("🚀 Starting Quantum Fraud Detection Web Application...")
    print("=" * 60)
    
    # Check if required files exist
    required_files = [
        "quantum_meta_model.py",
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
    print("🔧 Starting web server...")
    print("📱 Web interface will be available at: http://localhost:8000")
    print("🌐 Opening browser automatically...")
    
    # Open browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        # Start the web application
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Web application stopped by user")
    except Exception as e:
        print(f"❌ Error starting web application: {e}")

if __name__ == "__main__":
    main()