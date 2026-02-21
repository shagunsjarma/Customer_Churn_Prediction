import os
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from src.utils.logger import logging

class MLflowConfig:
    """MLflow configuration and utilities for the churn prediction project"""
    
    def __init__(self):
        # Set MLflow tracking URI - you can change this to a remote server
        # For local development, we'll use the default file store
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///mlruns")
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set experiment name
        self.experiment_name = "churn_prediction"
        
        # Initialize MLflow client
        self.client = MlflowClient()
        
        # Create or get experiment
        self.experiment_id = self._setup_experiment()
        
    def _setup_experiment(self):
        """Create or get the experiment"""
        try:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = self.client.create_experiment(self.experiment_name)
                logging.info(f"Created new experiment: {self.experiment_name} with ID: {experiment_id}")
            else:
                experiment_id = experiment.experiment_id
                logging.info(f"Using existing experiment: {self.experiment_name} with ID: {experiment_id}")
            return experiment_id
        except Exception as e:
            logging.error(f"Error setting up experiment: {str(e)}")
            # Fallback to default experiment
            return "0"
    
    def start_run(self, run_name=None):
        """Start a new MLflow run"""
        return mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name
        )
    
    def log_model_metrics(self, metrics):
        """Log model metrics to MLflow"""
        try:
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            logging.info(f"Logged metrics: {list(metrics.keys())}")
        except Exception as e:
            logging.error(f"Error logging metrics: {str(e)}")
    
    def log_model_parameters(self, params):
        """Log model parameters to MLflow"""
        try:
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            logging.info(f"Logged parameters: {list(params.keys())}")
        except Exception as e:
            logging.error(f"Error logging parameters: {str(e)}")
    
    def log_model(self, model, model_name, signature=None, input_example=None):
        """Log model to MLflow"""
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=model_name,
                signature=signature,
                input_example=input_example
            )
            logging.info(f"Logged model: {model_name}")
        except Exception as e:
            logging.error(f"Error logging model: {str(e)}")
    
    def log_preprocessor(self, preprocessor, preprocessor_name):
        """Log preprocessor to MLflow"""
        try:
            mlflow.sklearn.log_model(
                sk_model=preprocessor,
                artifact_path="preprocessor",
                registered_model_name=preprocessor_name
            )
            logging.info(f"Logged preprocessor: {preprocessor_name}")
        except Exception as e:
            logging.error(f"Error logging preprocessor: {str(e)}")
    
    def log_artifacts(self, artifacts_dir):
        """Log artifacts directory to MLflow"""
        try:
            mlflow.log_artifacts(artifacts_dir)
            logging.info(f"Logged artifacts from: {artifacts_dir}")
        except Exception as e:
            logging.error(f"Error logging artifacts: {str(e)}")
    
    def log_data_info(self, train_shape, test_shape, feature_names=None):
        """Log dataset information"""
        try:
            mlflow.log_param("train_samples", train_shape[0])
            mlflow.log_param("train_features", train_shape[1])
            mlflow.log_param("test_samples", test_shape[0])
            mlflow.log_param("test_features", test_shape[1])
            
            if feature_names:
                mlflow.log_param("feature_names", ",".join(feature_names))
            
            logging.info("Logged dataset information")
        except Exception as e:
            logging.error(f"Error logging data info: {str(e)}")
    
    def get_latest_model(self, model_name):
        """Get the latest version of a registered model"""
        try:
            latest_version = self.client.get_latest_versions(model_name, stages=["None"])
            if latest_version:
                return latest_version[0]
            return None
        except Exception as e:
            logging.error(f"Error getting latest model: {str(e)}")
            return None
    
    def load_model_for_prediction(self, model_name, version=None):
        """Load model for prediction"""
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            model = mlflow.sklearn.load_model(model_uri)
            logging.info(f"Loaded model: {model_name} version: {version or 'latest'}")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return None
