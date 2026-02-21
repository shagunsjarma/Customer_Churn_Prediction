#!/usr/bin/env python3
"""
MLflow Setup Script for Churn Prediction Project

This script sets up MLflow for the churn prediction project, including:
- Installing dependencies
- Setting up MLflow tracking
- Creating initial experiments
- Training and logging models
"""

import os
import sys
import subprocess
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from src.utils.mlflow_config import MLflowConfig
from src.utils.logger import logging

def install_dependencies():
    """Install required dependencies"""
    try:
        print("Installing MLflow dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def setup_mlflow_tracking():
    """Set up MLflow tracking"""
    try:
        print("Setting up MLflow tracking...")
        
        # Set tracking URI (default is local file store)
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        
        print(f"MLflow tracking URI set to: {tracking_uri}")
        
        # Create mlruns directory if it doesn't exist
        if tracking_uri.startswith("file://"):
            mlruns_dir = tracking_uri.replace("file://", "")
            os.makedirs(mlruns_dir, exist_ok=True)
            print(f"Created MLflow runs directory: {mlruns_dir}")
        
        return True
    except Exception as e:
        print(f"Error setting up MLflow tracking: {e}")
        return False

def create_experiment():
    """Create MLflow experiment"""
    try:
        print("Creating MLflow experiment...")
        mlflow_config = MLflowConfig()
        print(f"Experiment '{mlflow_config.experiment_name}' created/verified")
        return True
    except Exception as e:
        print(f"Error creating experiment: {e}")
        return False

def train_and_log_model():
    """Train model and log to MLflow"""
    try:
        print("Training model and logging to MLflow...")
        
        # Import training pipeline
        from src.models.trainer_pipeline import DataIngestion, DataTransformation, ModelTrainer
        
        # Run the training pipeline
        obj = DataIngestion()
        raw_data_path, train_data_path, test_data_path = obj.initiate_data_ingestion()
        
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        
        model_trainer = ModelTrainer()
        accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)
        
        print(f"Model training completed with accuracy: {accuracy}")
        print("Model logged to MLflow successfully!")
        return True
        
    except Exception as e:
        print(f"Error training and logging model: {e}")
        return False

def verify_setup():
    """Verify MLflow setup"""
    try:
        print("Verifying MLflow setup...")
        
        # Check if MLflow is accessible
        client = MlflowClient()
        experiments = client.search_experiments()
        
        print(f"Found {len(experiments)} experiments:")
        for exp in experiments:
            print(f"  - {exp.name} (ID: {exp.experiment_id})")
        
        # Check for registered models
        models = client.search_registered_models()
        print(f"Found {len(models)} registered models:")
        for model in models:
            print(f"  - {model.name}")
        
        print("MLflow setup verification completed!")
        return True
        
    except Exception as e:
        print(f"Error verifying setup: {e}")
        return False

def start_mlflow_ui():
    """Start MLflow UI"""
    try:
        print("Starting MLflow UI...")
        print("MLflow UI will be available at: http://localhost:5000")
        print("Press Ctrl+C to stop the UI")
        
        # Start MLflow UI
        subprocess.run(["mlflow", "ui", "--port", "5000"])
        
    except KeyboardInterrupt:
        print("\nMLflow UI stopped.")
    except Exception as e:
        print(f"Error starting MLflow UI: {e}")

def main():
    """Main setup function"""
    print("=== MLflow Setup for Churn Prediction Project ===\n")
    
    # Check if running as script
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "install":
            install_dependencies()
        elif command == "setup":
            setup_mlflow_tracking()
            create_experiment()
        elif command == "train":
            train_and_log_model()
        elif command == "verify":
            verify_setup()
        elif command == "ui":
            start_mlflow_ui()
        elif command == "all":
            print("Running complete setup...")
            if install_dependencies():
                if setup_mlflow_tracking():
                    if create_experiment():
                        if train_and_log_model():
                            verify_setup()
                            print("\n=== Setup Complete! ===")
                            print("You can now:")
                            print("1. Run 'python setup_mlflow.py ui' to start MLflow UI")
                            print("2. Use 'python mlflow_manage.py list-experiments' to see experiments")
                            print("3. Use the /predict_mlflow endpoint in your Flask app")
        else:
            print("Unknown command. Available commands:")
            print("  install - Install dependencies")
            print("  setup - Set up MLflow tracking")
            print("  train - Train and log model")
            print("  verify - Verify setup")
            print("  ui - Start MLflow UI")
            print("  all - Run complete setup")
    else:
        print("Usage: python setup_mlflow.py [command]")
        print("Commands:")
        print("  install - Install dependencies")
        print("  setup - Set up MLflow tracking")
        print("  train - Train and log model")
        print("  verify - Verify setup")
        print("  ui - Start MLflow UI")
        print("  all - Run complete setup")

if __name__ == "__main__":
    main()
