#!/usr/bin/env python3
"""
API Test Script for Customer Churn Prediction API
"""

import requests
import json
import time

# API Base URL
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\nTesting single prediction...")
    
    test_data = {
        "Age": 35,
        "Gender": "Female",
        "Tenure": 24,
        "Usage_Frequency": 15,
        "Support_Calls": 3,
        "Payment_Delay": 5,
        "Subscription_Type": "Standard",
        "Contract_Length": "Monthly",
        "Total_Spend": 850,
        "Last_Interaction": 10
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_model_info():
    """Test model info endpoint"""
    print("\nTesting model info...")
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_validation_errors():
    """Test validation error handling"""
    print("\nTesting validation errors...")
    
    # Test missing fields
    invalid_data = {
        "Age": 35,
        "Gender": "Female"
        # Missing other required fields
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 400
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all API tests"""
    print("=" * 50)
    print("Customer Churn Prediction API Tests")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Single Prediction", test_single_prediction),
        ("Model Info", test_model_info),
        ("Validation Errors", test_validation_errors)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
        
        time.sleep(1)  # Small delay between tests
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
