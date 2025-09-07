#!/usr/bin/env python3
"""
Test script for Plant Disease Detector application
"""

import os
import sys
import requests
import time
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import flask
        print(f"✅ Flask {flask.__version__}")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    return True

def test_file_structure():
    """Test if all required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'templates/base.html',
        'templates/index.html',
        'templates/training.html',
        'templates/analytics.html',
        'templates/about.html',
        'templates/contact.html',
        'static/css/style.css',
        'static/js/main.js',
        'nixpacks.toml',
        'Procfile',
        'railway.json',
        '.gitignore',
        'README.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def test_flask_app():
    """Test if Flask app can start"""
    print("\n🚀 Testing Flask application...")
    
    try:
        # Import the app
        sys.path.insert(0, os.getcwd())
        from app import app
        
        # Test app creation
        print("✅ Flask app created successfully")
        
        # Test routes
        with app.test_client() as client:
            # Test home route
            response = client.get('/')
            if response.status_code == 200:
                print("✅ Home route working")
            else:
                print(f"❌ Home route failed: {response.status_code}")
                return False
            
            # Test health check
            response = client.get('/health')
            if response.status_code == 200:
                print("✅ Health check working")
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
            
            # Test other routes
            routes = ['/training', '/analytics', '/about', '/contact']
            for route in routes:
                response = client.get(route)
                if response.status_code == 200:
                    print(f"✅ {route} route working")
                else:
                    print(f"❌ {route} route failed: {response.status_code}")
                    return False
        
        return True
        
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        return False

def test_model_files():
    """Test if model files exist"""
    print("\n🧠 Testing model files...")
    
    model_files = [
        'plant_disease_model.h5',
        'label_encoder.pkl'
    ]
    
    for file_path in model_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"⚠️  {file_path} - Not found (will be created on first training)")
    
    return True

def test_dataset():
    """Test if dataset exists"""
    print("\n📊 Testing dataset...")
    
    dataset_paths = ['Original Dataset', 'Augmented Dataset']
    
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            print(f"✅ {dataset_path} found")
            # Count images
            total_images = 0
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        total_images += 1
            print(f"   📸 {total_images:,} images found")
        else:
            print(f"⚠️  {dataset_path} - Not found")
    
    return True

def main():
    """Run all tests"""
    print("🌱 Plant Disease Detector - Application Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("File Structure Test", test_file_structure),
        ("Flask App Test", test_flask_app),
        ("Model Files Test", test_model_files),
        ("Dataset Test", test_dataset)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Application is ready to run.")
        print("\n🚀 To start the application:")
        print("   python app.py")
        print("\n🌐 Then open: http://localhost:5000")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
