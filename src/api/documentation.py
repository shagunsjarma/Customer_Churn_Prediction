"""
API Documentation for Customer Churn Prediction API
"""

API_DOCUMENTATION = {
    "title": "Customer Churn Prediction API",
    "version": "1.0.0",
    "description": "A machine learning API for predicting customer churn risk",
    "base_url": "/api/v1",
    "endpoints": {
        "health": {
            "path": "/health",
            "method": "GET",
            "description": "Check API and model health status",
            "response": {
                "200": {
                    "description": "API is healthy",
                    "example": {
                        "status": "healthy",
                        "model_loaded": True,
                        "preprocessor_loaded": True,
                        "api_version": "1.0.0",
                        "service": "Customer Churn Prediction API"
                    }
                },
                "503": {
                    "description": "API is unhealthy",
                    "example": {
                        "status": "unhealthy",
                        "model_loaded": False,
                        "preprocessor_loaded": False,
                        "api_version": "1.0.0",
                        "service": "Customer Churn Prediction API"
                    }
                }
            }
        },
        "predict": {
            "path": "/predict",
            "method": "POST",
            "description": "Predict churn risk for a single customer",
            "request_body": {
                "content_type": "application/json",
                "required_fields": [
                    "Age", "Gender", "Tenure", "Usage_Frequency", "Support_Calls",
                    "Payment_Delay", "Subscription_Type", "Contract_Length", 
                    "Total_Spend", "Last_Interaction"
                ],
                "example": {
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
            },
            "response": {
                "200": {
                    "description": "Successful prediction",
                    "example": {
                        "prediction": 1,
                        "probability": 0.85,
                        "churn_risk": "High",
                        "confidence": "Very High",
                        "recommendation": "Immediate intervention required. Consider retention offers and personal outreach.",
                        "input_data": {
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
                    }
                },
                "400": {
                    "description": "Bad request - validation error",
                    "example": {
                        "error": "Missing required fields: Age, Gender"
                    }
                },
                "500": {
                    "description": "Internal server error",
                    "example": {
                        "error": "An unexpected error occurred during prediction"
                    }
                }
            }
        },
        "predict_batch": {
            "path": "/predict/batch",
            "method": "POST",
            "description": "Predict churn risk for multiple customers",
            "request_body": {
                "content_type": "application/json",
                "example": {
                    "customers": [
                        {
                            "customer_id": "CUST001",
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
                        },
                        {
                            "customer_id": "CUST002",
                            "Age": 45,
                            "Gender": "Male",
                            "Tenure": 36,
                            "Usage_Frequency": 8,
                            "Support_Calls": 1,
                            "Payment_Delay": 2,
                            "Subscription_Type": "Premium",
                            "Contract_Length": "Annual",
                            "Total_Spend": 1200,
                            "Last_Interaction": 5
                        }
                    ]
                }
            },
            "response": {
                "200": {
                    "description": "Successful batch prediction",
                    "example": {
                        "results": [
                            {
                                "customer_id": "CUST001",
                                "prediction": 1,
                                "probability": 0.85,
                                "churn_risk": "High",
                                "confidence": "Very High"
                            },
                            {
                                "customer_id": "CUST002",
                                "prediction": 0,
                                "probability": 0.15,
                                "churn_risk": "Low",
                                "confidence": "Very High"
                            }
                        ],
                        "errors": [],
                        "total_processed": 2,
                        "total_errors": 0
                    }
                }
            }
        },
        "model_info": {
            "path": "/model/info",
            "method": "GET",
            "description": "Get information about the trained model",
            "response": {
                "200": {
                    "description": "Model information",
                    "example": {
                        "model_type": "RandomForestClassifier",
                        "preprocessor_type": "ColumnTransformer",
                        "features": [
                            "Age", "Gender", "Tenure", "Usage_Frequency", "Support_Calls",
                            "Payment_Delay", "Subscription_Type", "Contract_Length", 
                            "Total_Spend", "Last_Interaction"
                        ],
                        "feature_types": {
                            "categorical": ["Gender", "Contract_Length", "Subscription_Type"],
                            "numerical": ["Age", "Tenure", "Usage_Frequency", "Support_Calls", 
                                         "Payment_Delay", "Total_Spend", "Last_Interaction"]
                        },
                        "api_version": "1.0.0"
                    }
                }
            }
        },
        "feature_info": {
            "path": "/model/features",
            "method": "GET",
            "description": "Get detailed information about model features",
            "response": {
                "200": {
                    "description": "Feature information",
                    "example": {
                        "Age": {
                            "type": "numerical",
                            "description": "Customer age in years",
                            "range": "18-100",
                            "required": True
                        },
                        "Gender": {
                            "type": "categorical",
                            "description": "Customer gender",
                            "values": ["Male", "Female"],
                            "required": True
                        }
                    }
                }
            }
        }
    },
    "field_descriptions": {
        "Age": {
            "type": "numerical",
            "description": "Customer age in years",
            "range": "18-100",
            "required": True
        },
        "Gender": {
            "type": "categorical",
            "description": "Customer gender",
            "values": ["Male", "Female"],
            "required": True
        },
        "Tenure": {
            "type": "numerical",
            "description": "Number of months as customer",
            "range": "0-120",
            "required": True
        },
        "Usage_Frequency": {
            "type": "numerical",
            "description": "How often customer uses the service per month",
            "range": "0-50",
            "required": True
        },
        "Support_Calls": {
            "type": "numerical",
            "description": "Number of support calls made",
            "range": "0-50",
            "required": True
        },
        "Payment_Delay": {
            "type": "numerical",
            "description": "Average days of payment delay",
            "range": "0-90",
            "required": True
        },
        "Subscription_Type": {
            "type": "categorical",
            "description": "Type of subscription",
            "values": ["Basic", "Standard", "Premium"],
            "required": True
        },
        "Contract_Length": {
            "type": "categorical",
            "description": "Length of contract",
            "values": ["Monthly", "Quarterly", "Annual"],
            "required": True
        },
        "Total_Spend": {
            "type": "numerical",
            "description": "Total amount spent by customer",
            "range": "0-10000",
            "required": True
        },
        "Last_Interaction": {
            "type": "numerical",
            "description": "Days since last customer interaction",
            "range": "0-365",
            "required": True
        }
    },
    "response_codes": {
        "200": "Success",
        "400": "Bad Request - Invalid input data",
        "429": "Too Many Requests - Rate limit exceeded",
        "500": "Internal Server Error",
        "503": "Service Unavailable - Model not loaded"
    }
}

def get_api_docs():
    """
    Return API documentation
    """
    return API_DOCUMENTATION

def generate_curl_examples():
    """
    Generate cURL examples for API endpoints
    """
    examples = {
        "health_check": """
curl -X GET http://localhost:5000/api/v1/health
        """.strip(),
        
        "single_prediction": """
curl -X POST http://localhost:5000/api/v1/predict \\
  -H "Content-Type: application/json" \\
  -d '{
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
  }'
        """.strip(),
        
        "batch_prediction": """
curl -X POST http://localhost:5000/api/v1/predict/batch \\
  -H "Content-Type: application/json" \\
  -d '{
    "customers": [
      {
        "customer_id": "CUST001",
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
    ]
  }'
        """.strip(),
        
        "model_info": """
curl -X GET http://localhost:5000/api/v1/model/info
        """.strip(),
        
        "feature_info": """
curl -X GET http://localhost:5000/api/v1/model/features
        """.strip()
    }
    
    return examples
