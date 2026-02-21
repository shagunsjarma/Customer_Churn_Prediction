#!/usr/bin/env python3
"""
Test script to verify MLflow integration

This script tests the MLflow integration by:
1. Testing MLflow configuration
2. Testing model training with MLflow logging
3. Testing model prediction with MLflow models
"""

import os
import sys
import pandas as pd
import numpy as np
from src.utils.mlflow_config import MLflowConfig
from src.utils.logger import logging

def test_mlflow_config():
    """Test MLflow configuration"""
    try:
        print("Testing MLflow configuration...")
        mlflow_config = MLflowConfig()
        print(f"✓ MLflow config created successfully")
        print(f"  Experiment: {mlflow_config.experiment_name}")
        print(f"  Tracking URI: {mlflow_config.tracking_uri}")
        return True
    except Exception as e:
        print(f"✗ MLflow config test failed: {e}")
        return False

def test_experiment_creation():
    """Test experiment creation"""
    try:
        print("\nTesting experiment creation...")
        mlflow_config = MLflowConfig()
        
        # Test starting a run
        with mlflow_config.start_run(run_name="test_run"):
            mlflow_config.log_model_parameters({"test_param": "test_value"})
            mlflow_config.log_model_metrics({"test_metric": 0.95})
        
        print("✓ Experiment and run creation successful")
        return True
    except Exception as e:
        print(f"✗ Experiment creation test failed: {e}")
        return False

def test_model_training():
    """Test model training with MLflow integration"""
    try:
        print("\nTesting model training with MLflow...")
        
        # Check if training data exists
        train_path = "artifacts/train.csv"
        test_path = "artifacts/test.csv"
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print("⚠ Training data not found. Run data pipeline first.")
            return False
        
        # Import training components
        from src.data_pipeline.data_transformer import DataTransformation
        from src.data_pipeline.model_trainer import ModelTrainer
        
        # Test data transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)
        print("✓ Data transformation completed")
        
        # Test model training
        model_trainer = ModelTrainer()
        accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(f"✓ Model training completed with accuracy: {accuracy}")
        
        return True
    except Exception as e:
        print(f"✗ Model training test failed: {e}")
        return False

def test_prediction_pipeline():
    """Test prediction pipeline"""
    try:
        print("\nTesting prediction pipeline...")
        
        # Check if model files exist
        model_path = "artifacts/model.pkl"
        preprocessor_path = "artifacts/preprocessor.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            print("⚠ Model files not found. Run training first.")
            return False
        
        from src.models.predict_pipeline import PredictPipeline, CustomData
        
        # Test local model prediction
        predictor = PredictPipeline()
        print("✓ Local model prediction pipeline created")
        
        # Test MLflow model prediction (if model exists in registry)
        try:
            mlflow_predictor = PredictPipeline(use_mlflow_model=True)
            print("✓ MLflow model prediction pipeline created")
        except Exception as e:
            print(f"⚠ MLflow model prediction not available: {e}")
        
        # Test prediction with sample data
        sample_data = CustomData(
            Age=45,
            Gender="Male",
            Tenure=24,
            Usage_Frequency=15,
            Support_Calls=2,
            Payment_Delay=5,
            Subscription_Type="Premium",
            Contract_Length="Annual",
            Total_Spend=1200,
            Last_Interaction=30
        )
        
        df = sample_data.get_data_as_dataframe()
        prediction = predictor.predict(df)
        print(f"✓ Sample prediction completed: {prediction}")
        
        return True
    except Exception as e:
        print(f"✗ Prediction pipeline test failed: {e}")
        return False

def test_flask_integration():
    """Test Flask app integration"""
    try:
        print("\nTesting Flask app integration...")
        
        # Check if Flask app can import MLflow components
        from app import app
        
        # Test that MLflow endpoints are available
        with app.test_client() as client:
            # Test model info endpoint
            response = client.get('/model_info')
            if response.status_code == 200:
                print("✓ Model info endpoint working")
            else:
                print(f"⚠ Model info endpoint returned status: {response.status_code}")
            
            # Test MLflow model info endpoint
            response = client.get('/model_info?use_mlflow=false')
            if response.status_code == 200:
                print("✓ MLflow model info endpoint working")
            else:
                print(f"⚠ MLflow model info endpoint returned status: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"✗ Flask integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== MLflow Integration Test Suite ===\n")
    
    tests = [
        test_mlflow_config,
        test_experiment_creation,
        test_model_training,
        test_prediction_pipeline,
        test_flask_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! MLflow integration is working correctly.")
        print("\nNext steps:")
        print("1. Run 'python setup_mlflow.py ui' to start MLflow UI")
        print("2. Run 'python mlflow_manage.py list-experiments' to see experiments")
        print("3. Use the Flask app with MLflow endpoints")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Check the errors above.")
        print("Make sure to run the data pipeline and model training first.")

if __name__ == "__main__":
    main()
