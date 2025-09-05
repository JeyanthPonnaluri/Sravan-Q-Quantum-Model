#!/usr/bin/env python3
"""
Test script to check available Gemini models and fix API issues
"""

import google.generativeai as genai
import os

def test_gemini_models():
    """Test available Gemini models"""
    
    # Configure API key
    api_key = "AIzaSyCjKN-Cmn4PDcJoM_rV--5idHdD6NxP3tE"
    genai.configure(api_key=api_key)
    
    print("🔍 Testing Gemini API Models")
    print("=" * 50)
    
    try:
        # List all available models
        print("📋 Listing all available models...")
        models = genai.list_models()
        
        available_models = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
                print(f"✅ {model.name}")
        
        if not available_models:
            print("❌ No models support generateContent")
            return False
        
        # Test each available model
        print(f"\n🧪 Testing models with simple prompt...")
        
        test_prompt = "Analyze this transaction: Amount ₹50000 at 2 AM. Is this suspicious? Answer in one sentence."
        
        for model_name in available_models[:3]:  # Test first 3 models
            try:
                print(f"\n🔬 Testing {model_name}...")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(test_prompt)
                print(f"✅ {model_name}: {response.text[:100]}...")
                return True  # If we get here, at least one model works
                
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
                continue
        
        print("❌ All models failed to generate content")
        return False
        
    except Exception as e:
        print(f"❌ Error accessing Gemini API: {e}")
        return False

def test_specific_models():
    """Test specific model names"""
    
    api_key = "AIzaSyCjKN-Cmn4PDcJoM_rV--5idHdD6NxP3tE"
    genai.configure(api_key=api_key)
    
    print(f"\n🎯 Testing specific model names...")
    
    model_names_to_test = [
        'gemini-1.5-pro-latest',
        'gemini-1.5-flash-latest', 
        'gemini-1.5-pro',
        'gemini-1.5-flash',
        'gemini-pro',
        'models/gemini-1.5-pro-latest',
        'models/gemini-1.5-flash-latest'
    ]
    
    test_prompt = "Rate fraud risk 1-10 for: ₹100000 at 3AM weekend. Reply with just number."
    
    for model_name in model_names_to_test:
        try:
            print(f"🔬 Testing {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(test_prompt)
            print(f"✅ {model_name} works! Response: {response.text.strip()}")
            return model_name  # Return the first working model
            
        except Exception as e:
            print(f"❌ {model_name}: {str(e)[:100]}...")
            continue
    
    return None

if __name__ == "__main__":
    print("🚀 Gemini API Model Testing")
    print("=" * 60)
    
    # Test 1: List available models
    success = test_gemini_models()
    
    # Test 2: Test specific model names
    working_model = test_specific_models()
    
    print(f"\n📊 RESULTS:")
    if working_model:
        print(f"✅ Working model found: {working_model}")
        print(f"💡 Update gemini_logical_model.py to use: {working_model}")
    else:
        print("❌ No working models found")
        print("💡 Check API key and internet connection")
    
    print(f"\n🔧 RECOMMENDED FIX:")
    if working_model:
        print(f"Replace 'gemini-pro' with '{working_model}' in gemini_logical_model.py")
    else:
        print("Use fallback mode without Gemini API")