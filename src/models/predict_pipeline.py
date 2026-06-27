import os
import sys
import pandas as pd
import numpy as np
from src.utils.save_model import load_object
from src.utils.exception import CustomException
from src.utils.mlflow_config import MLflowConfig
from datetime import datetime

class PredictPipeline:
    def __init__(self, use_mlflow_model=False, model_version=None):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.use_mlflow_model = use_mlflow_model
        self.model_version = model_version
        self.mlflow_config = MLflowConfig()
        
        # Load model and preprocessor
        self._load_models()
        
    def _load_models(self):
        """Load model and preprocessor from local files or MLflow"""
        try:
            if self.use_mlflow_model:
                # Load from MLflow
                self.model = self.mlflow_config.load_model_for_prediction(
                    "churn_prediction_model", 
                    version=self.model_version
                )
                if self.model is None:
                    raise CustomException("Failed to load model from MLflow", sys)
            else:
                # Load from local files (existing functionality)
                self.model = load_object(self.model_path)
            
            # Preprocessor is still loaded locally for now
            self.preprocessor = load_object(self.preprocessor_path)
            
        except Exception as e:
            raise CustomException(f"Error loading models: {str(e)}", sys)
        
    def predict(self, features):
        try:
            if isinstance(features, dict):
                df = pd.DataFrame([features])
            elif isinstance(features, pd.DataFrame):
                df = features.copy()
            else:
                raise CustomException("Input features should be a dictionary or a pandas DataFrame", sys)
            
            # Transform features
            data_transformed = self.preprocessor.transform(df)
            
            # Make prediction
            preds = self.model.predict(data_transformed)
            
            # Log prediction to MLflow if enabled
            if self.use_mlflow_model:
                self._log_prediction(df, preds)
            
            return preds
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_with_probability(self, features):
        """Make prediction with probability scores"""
        try:
            if isinstance(features, dict):
                df = pd.DataFrame([features])
            elif isinstance(features, pd.DataFrame):
                df = features.copy()
            else:
                raise CustomException("Input features should be a dictionary or a pandas DataFrame", sys)
            
            # Transform features
            data_transformed = self.preprocessor.transform(df)
            
            # Make prediction
            preds = self.model.predict(data_transformed)
            
            # Get probabilities if available
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(data_transformed)
            
            # Log prediction to MLflow if enabled
            if self.use_mlflow_model:
                self._log_prediction(df, preds, probabilities)
            
            return preds, probabilities
        except Exception as e:
            raise CustomException(e, sys)
    
    def _log_prediction(self, features_df, predictions, probabilities=None):
        """Log prediction details to Database for drift detection"""
        try:
            from src.monitoring.database import SessionLocal, PredictionLog
            
            db = SessionLocal()
            try:
                for i in range(len(predictions)):
                    row = features_df.iloc[i]
                    prob = float(probabilities[i, 1]) if probabilities is not None else None
                    
                    db_log = PredictionLog(
                        Age=float(row.get('Age', 0)),
                        Gender=str(row.get('Gender', '')),
                        Tenure=float(row.get('Tenure', 0)),
                        Usage_Frequency=float(row.get('Usage Frequency', 0)),
                        Support_Calls=float(row.get('Support Calls', 0)),
                        Payment_Delay=float(row.get('Payment Delay', 0)),
                        Subscription_Type=str(row.get('Subscription Type', '')),
                        Contract_Length=str(row.get('Contract Length', '')),
                        Total_Spend=float(row.get('Total Spend', 0)),
                        Last_Interaction=float(row.get('Last Interaction', 0)),
                        model_version=self.model_version,
                        prediction=int(predictions[i]),
                        probability=prob
                    )
                    db.add(db_log)
                db.commit()
            finally:
                db.close()
                
            prediction_info = {
                "prediction_count": len(predictions),
                "prediction_distribution": {
                    "churn_0": int(np.sum(predictions == 0)),
                    "churn_1": int(np.sum(predictions == 1))
                }
            }
            if probabilities is not None:
                prediction_info["avg_churn_probability"] = float(np.mean(probabilities[:, 1]))
            print(f"Prediction logged to Database: {prediction_info}")
            
        except Exception as e:
            # Don't raise exception for logging failures
            print(f"Warning: Could not log prediction to Database: {str(e)}")
        
class CustomData:
    def __init__(self, 
                 Age: float,
                 Gender: str,
                 Tenure: float,
                 Usage_Frequency: float,
                 Support_Calls: float,
                 Payment_Delay: float,
                 Subscription_Type: str,
                 Contract_Length: str,
                 Total_Spend: float,
                 Last_Interaction: float
                 ):  
        self.Age = Age
        self.Gender = Gender
        self.Tenure = Tenure
        self.Usage_Frequency = Usage_Frequency
        self.Support_Calls = Support_Calls
        self.Payment_Delay = Payment_Delay
        self.Subscription_Type = Subscription_Type
        self.Contract_Length = Contract_Length
        self.Total_Spend = Total_Spend
        self.Last_Interaction = Last_Interaction
        
    def get_data_as_dataframe(self):
        try:
            data = {
                "Age": [self.Age],
                "Gender": [self.Gender],
                "Tenure": [self.Tenure],
                "Usage Frequency": [self.Usage_Frequency],
                "Support Calls": [self.Support_Calls],
                "Payment Delay": [self.Payment_Delay],
                "Subscription Type": [self.Subscription_Type],
                "Contract Length": [self.Contract_Length],
                "Total Spend": [self.Total_Spend],
                "Last Interaction": [self.Last_Interaction]
            }
            df = pd.DataFrame(data) 
            return df
        except Exception as e:
            raise CustomException(e, sys)              