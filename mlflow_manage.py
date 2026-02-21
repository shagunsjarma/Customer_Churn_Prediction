#!/usr/bin/env python3
"""
MLflow Management Script for Churn Prediction Project

This script provides utilities to manage MLflow experiments, models, and runs.
"""

import os
import sys
import argparse
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from src.utils.mlflow_config import MLflowConfig
from src.utils.logger import logging

def list_experiments():
    """List all MLflow experiments"""
    try:
        client = MlflowClient()
        experiments = client.search_experiments()
        
        print("\n=== MLflow Experiments ===")
        for exp in experiments:
            print(f"ID: {exp.experiment_id}")
            print(f"Name: {exp.name}")
            print(f"Artifact Location: {exp.artifact_location}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error listing experiments: {str(e)}")

def list_runs(experiment_name="churn_prediction"):
    """List runs for a specific experiment"""
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found")
            return
            
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        
        print(f"\n=== Runs for Experiment: {experiment_name} ===")
        for run in runs:
            print(f"Run ID: {run.info.run_id}")
            print(f"Status: {run.info.status}")
            print(f"Start Time: {run.info.start_time}")
            print(f"End Time: {run.info.end_time}")
            
            # Print metrics
            if run.data.metrics:
                print("Metrics:")
                for key, value in run.data.metrics.items():
                    print(f"  {key}: {value}")
            
            # Print parameters
            if run.data.params:
                print("Parameters:")
                for key, value in run.data.params.items():
                    print(f"  {key}: {value}")
            
            print("-" * 50)
            
    except Exception as e:
        print(f"Error listing runs: {str(e)}")

def list_models():
    """List registered models"""
    try:
        client = MlflowClient()
        models = client.search_registered_models()
        
        print("\n=== Registered Models ===") 
        for model in models:
            print(f"Name: {model.name}")
            print(f"Latest Version: {model.latest_versions[0].version if model.latest_versions else 'None'}")
            print(f"Description: {model.description}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error listing models: {str(e)}")

def get_model_info(model_name="churn_prediction_model"):
    """Get information about a specific model"""
    try:
        client = MlflowClient()
        
        # Get model versions
        versions = client.get_latest_versions(model_name)
        
        print(f"\n=== Model Information: {model_name} ===")
        
        if not versions:
            print("No versions found for this model")
            return
            
        for version in versions:
            print(f"Version: {version.version}")
            print(f"Stage: {version.current_stage}")
            print(f"Run ID: {version.run_id}")
            print(f"Creation Time: {version.creation_timestamp}")
            print(f"Description: {version.description}")
            print("-" * 50)
            
        # Get the latest model details
        latest_version = versions[0]
        run = client.get_run(latest_version.run_id)
        
        print("Latest Model Metrics:")
        for key, value in run.data.metrics.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error getting model info: {str(e)}")

def load_model_for_testing(model_name="churn_prediction_model", version=None):
    """Load a model for testing"""
    try:
        mlflow_config = MLflowConfig()
        
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/latest"
            
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Successfully loaded model: {model_uri}")
        print(f"Model type: {type(model).__name__}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def delete_run(run_id):
    """Delete a specific run"""
    try:
        client = MlflowClient()
        client.delete_run(run_id)
        print(f"Successfully deleted run: {run_id}")
        
    except Exception as e:
        print(f"Error deleting run: {str(e)}")

def export_model(model_name="churn_prediction_model", version=None, output_dir="./exported_models"):
    """Export a model for deployment"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        if version:
            model_uri = f"models:/{model_name}/{version}"
            output_path = os.path.join(output_dir, f"{model_name}_v{version}")
        else:
            model_uri = f"models:/{model_name}/latest"
            output_path = os.path.join(output_dir, f"{model_name}_latest")
            
        mlflow.sklearn.save_model(
            sk_model=mlflow.sklearn.load_model(model_uri),
            path=output_path
        )
        
        print(f"Model exported to: {output_path}")
        
    except Exception as e:
        print(f"Error exporting model: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="MLflow Management for Churn Prediction")
    parser.add_argument("command", choices=[
        "list-experiments", "list-runs", "list-models", "model-info", 
        "load-model", "delete-run", "export-model", "start-ui"
    ], help="Command to execute")
    
    parser.add_argument("--experiment", default="churn_prediction", help="Experiment name")
    parser.add_argument("--model", default="churn_prediction_model", help="Model name")
    parser.add_argument("--version", help="Model version")
    parser.add_argument("--run-id", help="Run ID for deletion")
    parser.add_argument("--output-dir", default="./exported_models", help="Output directory for exports")
    parser.add_argument("--port", default=5001, help="Port for MLflow UI")
    
    args = parser.parse_args()
    
    if args.command == "list-experiments":
        list_experiments()
    elif args.command == "list-runs":
        list_runs(args.experiment)
    elif args.command == "list-models":
        list_models()
    elif args.command == "model-info":
        get_model_info(args.model)
    elif args.command == "load-model":
        load_model_for_testing(args.model, args.version)
    elif args.command == "delete-run":
        if not args.run_id:
            print("Error: --run-id is required for delete-run command")
            return
        delete_run(args.run_id)
    elif args.command == "export-model":
        export_model(args.model, args.version, args.output_dir)
    elif args.command == "start-ui":
        print(f"Starting MLflow UI on port {args.port}")
        print(f"Open http://localhost:{args.port} in your browser")
        os.system(f"mlflow ui --port {args.port}")

if __name__ == "__main__":
    main()
