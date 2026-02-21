import os
import sys
from dataclasses import dataclass
from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.save_model import save_object, load_object
from src.utils.mlflow_config import MLflowConfig
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
from datetime import datetime

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.mlflow_config = MLflowConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            # Define models with hyperparameters for MLflow tracking
            models = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10
                ),
                "Decision Tree": DecisionTreeClassifier(
                    random_state=42,
                    max_depth=10
                ),
                "Logistic Regression": LogisticRegression(
                    random_state=42,
                    max_iter=1000
                ),
                "XGB Classifier": XGBClassifier(
                    random_state=42,
                    max_depth=6,
                    learning_rate=0.1
                ),
                "AdaBoost Classifier": AdaBoostClassifier(
                    random_state=42,
                    n_estimators=100
                )
            }
            
            # Start MLflow run
            with self.mlflow_config.start_run(run_name=f"churn_prediction_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                
                # Log dataset information
                self.mlflow_config.log_data_info(
                    train_shape=X_train.shape,
                    test_shape=X_test.shape,
                    feature_names=[f"feature_{i}" for i in range(X_train.shape[1])]
                )
                
                # Log training parameters
                training_params = {
                    "train_samples": X_train.shape[0],
                    "test_samples": X_test.shape[0],
                    "n_features": X_train.shape[1],
                    "target_distribution": f"Churn: {np.sum(y_train)}/{len(y_train)} ({np.mean(y_train):.2%})"
                }
                self.mlflow_config.log_model_parameters(training_params)
                
                model_report = {}
                best_model_name = None
                best_model = None
                best_metrics = {}
                
                # Train and evaluate each model
                for model_name, model in models.items():
                    logging.info(f"Training {model_name}...")
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    metrics = {
                        f"{model_name}_accuracy": accuracy,
                        f"{model_name}_precision": precision,
                        f"{model_name}_recall": recall,
                        f"{model_name}_f1_score": f1
                    }
                    
                    # Add ROC AUC if probabilities are available
                    if y_pred_proba is not None:
                        try:
                            roc_auc = roc_auc_score(y_test, y_pred_proba)
                            metrics[f"{model_name}_roc_auc"] = roc_auc
                        except:
                            pass
                    
                    # Log metrics for this model
                    self.mlflow_config.log_model_metrics(metrics)
                    
                    # Log model parameters
                    model_params = self._extract_model_params(model, model_name)
                    self.mlflow_config.log_model_parameters(model_params)
                    
                    model_report[model_name] = accuracy
                    
                    # Track best model
                    if best_model_name is None or accuracy > model_report[best_model_name]:
                        best_model_name = model_name
                        best_model = model
                        best_metrics = {
                            "best_model_accuracy": accuracy,
                            "best_model_precision": precision,
                            "best_model_recall": recall,
                            "best_model_f1_score": f1
                        }
                        if y_pred_proba is not None:
                            try:
                                roc_auc = roc_auc_score(y_test, y_pred_proba)
                                best_metrics["best_model_roc_auc"] = roc_auc
                            except:
                                pass
                
                # Log best model metrics
                self.mlflow_config.log_model_metrics(best_metrics)
                
                logging.info(f"Best model found: {best_model_name} with accuracy: {model_report[best_model_name]}")
                
                if model_report[best_model_name] < 0.5:
                    logging.warning(f"Best model accuracy {model_report[best_model_name]} is below 50%, but continuing with training")
                
                # Save model locally (existing functionality)
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )
                
                # Log model to MLflow
                self.mlflow_config.log_model(
                    model=best_model,
                    model_name="churn_prediction_model"
                )
                
                # Log model performance summary
                performance_summary = {
                    "model_comparison": model_report,
                    "best_model": best_model_name,
                    "best_accuracy": model_report[best_model_name]
                }
                self.mlflow_config.log_model_parameters(performance_summary)
                
                # Final evaluation
                predicted = best_model.predict(X_test)
                final_score = accuracy_score(y_test, predicted)
                
                logging.info(f"Model training completed. Final accuracy: {final_score}")
                return final_score
                
        except Exception as e:
            raise CustomException(e, sys)
    
    def _extract_model_params(self, model, model_name):
        """Extract parameters from model for MLflow logging"""
        params = {"model_type": model_name}
        
        if hasattr(model, 'get_params'):
            model_params = model.get_params()
            for key, value in model_params.items():
                if isinstance(value, (int, float, str, bool)):
                    params[f"{model_name}_{key}"] = value
        
        return params