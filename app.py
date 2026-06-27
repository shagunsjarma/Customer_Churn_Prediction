import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

load_dotenv()
from src.models.predict_pipeline import PredictPipeline, CustomData
from src.utils.exception import CustomException
from src.utils.logger import logging
import traceback

app = FastAPI(title="Churn Prediction API", version="1.0.0")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Pydantic models for validation
class ChurnPredictionRequest(BaseModel):
    Age: float = Field(..., ge=18, le=100, description="Age of the customer")
    Gender: str = Field(..., description="Male or Female")
    Tenure: float = Field(..., ge=0, le=120, description="Tenure in months")
    Usage_Frequency: float = Field(..., ge=0, le=50, description="Usage frequency")
    Support_Calls: float = Field(..., ge=0, le=50, description="Number of support calls")
    Payment_Delay: float = Field(..., ge=0, le=90, description="Payment delay in days")
    Subscription_Type: str = Field(..., description="Basic, Standard, or Premium")
    Contract_Length: str = Field(..., description="Monthly, Quarterly, or Annual")
    Total_Spend: float = Field(..., ge=0, le=10000, description="Total spend")
    Last_Interaction: float = Field(..., ge=0, le=365, description="Last interaction in days")
    model_version: Optional[str] = Field(None, description="Optional MLflow model version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "Age": 35.0,
                "Gender": "Female",
                "Tenure": 24.0,
                "Usage_Frequency": 15.0,
                "Support_Calls": 3.0,
                "Payment_Delay": 5.0,
                "Subscription_Type": "Standard",
                "Contract_Length": "Monthly",
                "Total_Spend": 850.0,
                "Last_Interaction": 10.0,
                "model_version": None
            }
        }

def get_churn_message(prediction: int, probability: float) -> str:
    """
    Generate appropriate message based on prediction and probability
    """
    if prediction == 1:
        if probability > 0.8:
            return "This customer has a very high risk of churning. Immediate intervention required."
        elif probability > 0.6:
            return "This customer has a high risk of churning. Proactive retention efforts recommended."
        else:
            return "This customer shows some signs of potential churn. Monitor closely."
    else:
        if probability < 0.3:
            return "This customer has a very low risk of churning. Continue current engagement strategy."
        elif probability < 0.5:
            return "This customer has a low risk of churning. Maintain current service levels."
        else:
            return "This customer shows moderate engagement. Consider upselling opportunities."

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Mount the static directory for CSS and JS
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def home():
    """
    Serve the frontend dashboard
    """
    return FileResponse("frontend/index.html")

@app.post("/predict")
async def predict(data: ChurnPredictionRequest):
    """
    Prediction endpoint that takes customer data and returns churn prediction
    """
    try:
        # Create CustomData object
        custom_data = CustomData(
            Age=data.Age,
            Gender=data.Gender,
            Tenure=data.Tenure,
            Usage_Frequency=data.Usage_Frequency,
            Support_Calls=data.Support_Calls,
            Payment_Delay=data.Payment_Delay,
            Subscription_Type=data.Subscription_Type,
            Contract_Length=data.Contract_Length,
            Total_Spend=data.Total_Spend,
            Last_Interaction=data.Last_Interaction
        )
        
        # Get data as DataFrame
        df = custom_data.get_data_as_dataframe()
        
        import random
        
        # A/B Testing Logic: Split traffic 90/10
        traffic_roll = random.random()
        is_staging = traffic_roll >= 0.9
        
        # Initialize prediction pipeline
        # In a real scenario, staging might load a different model file or MLflow version
        # For now, we simulate this by passing a model_version flag
        model_version = "staging" if is_staging else "production"
        predict_pipeline = PredictPipeline(model_version=model_version)
        
        # Make prediction
        prediction = predict_pipeline.predict(df)
        
        # Get prediction probability if available
        try:
            model = predict_pipeline.model
            preprocessor = predict_pipeline.preprocessor
            
            if hasattr(model, 'predict_proba'):
                data_transformed = preprocessor.transform(df)
                probabilities = model.predict_proba(data_transformed)
                probability = float(probabilities[0][1])  # Probability of churn (class 1)
            else:
                probability = 0.8 if prediction[0] == 1 else 0.2
        except Exception as e:
            logging.warning(f"Could not get prediction probabilities: {str(e)}")
            probability = 0.8 if prediction[0] == 1 else 0.2
        
        # Prepare response
        result = {
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'churn_risk': 'High' if prediction[0] == 1 else 'Low',
            'message': get_churn_message(prediction[0], probability),
            'model_variant': model_version
        }
        
        logging.info(f"Prediction made: {result}")
        return result
        
    except CustomException as e:
        logging.error(f"Custom exception in predict route: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error in predict route: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during prediction")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        artifacts_dir = 'artifacts'
        model_path = os.path.join(artifacts_dir, 'model.pkl')
        preprocessor_path = os.path.join(artifacts_dir, 'preprocessor.pkl')
        
        model_exists = os.path.exists(model_path)
        preprocessor_exists = os.path.exists(preprocessor_path)
        
        if model_exists and preprocessor_exists:
            return {
                'status': 'healthy',
                'model_loaded': True,
                'preprocessor_loaded': True
            }
        else:
            raise HTTPException(status_code=503, detail={
                'status': 'unhealthy',
                'model_loaded': model_exists,
                'preprocessor_loaded': preprocessor_exists,
                'message': 'Model or preprocessor files not found'
            })
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during health check")

@app.get("/model_info")
async def model_info(use_mlflow: bool = False, version: Optional[str] = None):
    """
    Get information about the trained model
    """
    try:
        if use_mlflow:
            predict_pipeline = PredictPipeline(use_mlflow_model=True, model_version=version)
        else:
            predict_pipeline = PredictPipeline()
        
        model_info = {
            'model_type': type(predict_pipeline.model).__name__,
            'preprocessor_type': type(predict_pipeline.preprocessor).__name__,
            'model_source': 'MLflow' if use_mlflow else 'Local',
            'model_version': version if use_mlflow else 'N/A',
            'features': list(ChurnPredictionRequest.model_fields.keys())[:-1] # Exclude model_version
        }
        
        return model_info
    except Exception as e:
        logging.error(f"Model info error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while fetching model info")

@app.post("/predict_mlflow")
async def predict_mlflow(data: ChurnPredictionRequest):
    """
    Prediction endpoint using MLflow model
    """
    try:
        custom_data = CustomData(
            Age=data.Age,
            Gender=data.Gender,
            Tenure=data.Tenure,
            Usage_Frequency=data.Usage_Frequency,
            Support_Calls=data.Support_Calls,
            Payment_Delay=data.Payment_Delay,
            Subscription_Type=data.Subscription_Type,
            Contract_Length=data.Contract_Length,
            Total_Spend=data.Total_Spend,
            Last_Interaction=data.Last_Interaction
        )
        
        df = custom_data.get_data_as_dataframe()
        
        predict_pipeline = PredictPipeline(use_mlflow_model=True, model_version=data.model_version)
        
        prediction, probabilities = predict_pipeline.predict_with_probability(df)
        
        if probabilities is not None:
            probability = float(probabilities[0][1])
        else:
            probability = 0.8 if prediction[0] == 1 else 0.2
        
        result = {
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'churn_risk': 'High' if prediction[0] == 1 else 'Low',
            'message': get_churn_message(prediction[0], probability),
            'model_source': 'MLflow',
            'model_version': data.model_version or 'latest'
        }
        
        logging.info(f"MLflow prediction made: {result}")
        return result
        
    except CustomException as e:
        logging.error(f"Custom exception in MLflow predict route: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error in MLflow predict route: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during prediction")
