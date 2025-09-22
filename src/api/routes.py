from flask import Blueprint, request, jsonify
import os
import sys
import pandas as pd
import numpy as np
from src.models.predict_pipeline import PredictPipeline, CustomData
from src.utils.exception import CustomException
from src.utils.logger import logging
import traceback

# Create Blueprint for API routes
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for API
    Returns the status of the API and model availability
    """
    try:
        # Check if model files exist
        model_path = os.path.join('artifacts', 'model.pkl')
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        
        model_exists = os.path.exists(model_path)
        preprocessor_exists = os.path.exists(preprocessor_path)
        
        status = "healthy" if (model_exists and preprocessor_exists) else "unhealthy"
        
        return jsonify({
            'status': status,
            'model_loaded': model_exists,
            'preprocessor_loaded': preprocessor_exists,
            'api_version': '1.0.0',
            'service': 'Customer Churn Prediction API'
        }), 200 if status == "healthy" else 503
        
    except Exception as e:
        logging.error(f"Health check error: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@api_bp.route('/predict', methods=['POST'])
def predict_churn():
    """
    Predict customer churn risk
    
    Expected JSON payload:
    {
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
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = [
            'Age', 'Gender', 'Tenure', 'Usage_Frequency', 'Support_Calls',
            'Payment_Delay', 'Subscription_Type', 'Contract_Length', 
            'Total_Spend', 'Last_Interaction'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'required_fields': required_fields
            }), 400
        
        # Validate data types and ranges
        validation_errors = validate_input_data(data)
        if validation_errors:
            return jsonify({'error': validation_errors}), 400
        
        # Create CustomData object
        custom_data = CustomData(
            Age=float(data['Age']),
            Gender=str(data['Gender']),
            Tenure=float(data['Tenure']),
            Usage_Frequency=float(data['Usage_Frequency']),
            Support_Calls=float(data['Support_Calls']),
            Payment_Delay=float(data['Payment_Delay']),
            Subscription_Type=str(data['Subscription_Type']),
            Contract_Length=str(data['Contract_Length']),
            Total_Spend=float(data['Total_Spend']),
            Last_Interaction=float(data['Last_Interaction'])
        )
        
        # Get data as DataFrame
        df = custom_data.get_data_as_dataframe()
        
        # Initialize prediction pipeline
        predict_pipeline = PredictPipeline()
        
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
                # If no predict_proba, use a default probability based on prediction
                probability = 0.8 if prediction[0] == 1 else 0.2
        except Exception as e:
            logging.warning(f"Could not get prediction probabilities: {str(e)}")
            probability = 0.8 if prediction[0] == 1 else 0.2
        
        # Prepare response
        result = {
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'churn_risk': 'High' if prediction[0] == 1 else 'Low',
            'confidence': get_confidence_level(probability),
            'recommendation': get_recommendation(prediction[0], probability),
            'input_data': data
        }
        
        logging.info(f"Prediction made: {result}")
        return jsonify(result), 200
        
    except CustomException as e:
        logging.error(f"Custom exception in predict route: {str(e)}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logging.error(f"Unexpected error in predict route: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'An unexpected error occurred during prediction'}), 500

@api_bp.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict churn for multiple customers at once
    
    Expected JSON payload:
    {
        "customers": [
            {
                "customer_id": "CUST001",
                "Age": 35,
                "Gender": "Female",
                ...
            },
            {
                "customer_id": "CUST002",
                "Age": 45,
                "Gender": "Male",
                ...
            }
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'customers' not in data:
            return jsonify({'error': 'No customers data provided'}), 400
        
        customers = data['customers']
        if not isinstance(customers, list) or len(customers) == 0:
            return jsonify({'error': 'customers must be a non-empty list'}), 400
        
        if len(customers) > 100:
            return jsonify({'error': 'Maximum 100 customers allowed per batch'}), 400
        
        results = []
        errors = []
        
        for i, customer in enumerate(customers):
            try:
                # Validate customer data
                required_fields = [
                    'Age', 'Gender', 'Tenure', 'Usage_Frequency', 'Support_Calls',
                    'Payment_Delay', 'Subscription_Type', 'Contract_Length', 
                    'Total_Spend', 'Last_Interaction'
                ]
                
                missing_fields = [field for field in required_fields if field not in customer]
                if missing_fields:
                    errors.append({
                        'index': i,
                        'customer_id': customer.get('customer_id', f'customer_{i}'),
                        'error': f'Missing required fields: {", ".join(missing_fields)}'
                    })
                    continue
                
                # Create CustomData object
                custom_data = CustomData(
                    Age=float(customer['Age']),
                    Gender=str(customer['Gender']),
                    Tenure=float(customer['Tenure']),
                    Usage_Frequency=float(customer['Usage_Frequency']),
                    Support_Calls=float(customer['Support_Calls']),
                    Payment_Delay=float(customer['Payment_Delay']),
                    Subscription_Type=str(customer['Subscription_Type']),
                    Contract_Length=str(customer['Contract_Length']),
                    Total_Spend=float(customer['Total_Spend']),
                    Last_Interaction=float(customer['Last_Interaction'])
                )
                
                # Get data as DataFrame
                df = custom_data.get_data_as_dataframe()
                
                # Initialize prediction pipeline
                predict_pipeline = PredictPipeline()
                
                # Make prediction
                prediction = predict_pipeline.predict(df)
                
                # Get probability
                try:
                    model = predict_pipeline.model
                    preprocessor = predict_pipeline.preprocessor
                    
                    if hasattr(model, 'predict_proba'):
                        data_transformed = preprocessor.transform(df)
                        probabilities = model.predict_proba(data_transformed)
                        probability = float(probabilities[0][1])
                    else:
                        probability = 0.8 if prediction[0] == 1 else 0.2
                except:
                    probability = 0.8 if prediction[0] == 1 else 0.2
                
                results.append({
                    'customer_id': customer.get('customer_id', f'customer_{i}'),
                    'prediction': int(prediction[0]),
                    'probability': float(probability),
                    'churn_risk': 'High' if prediction[0] == 1 else 'Low',
                    'confidence': get_confidence_level(probability)
                })
                
            except Exception as e:
                errors.append({
                    'index': i,
                    'customer_id': customer.get('customer_id', f'customer_{i}'),
                    'error': str(e)
                })
        
        return jsonify({
            'results': results,
            'errors': errors,
            'total_processed': len(results),
            'total_errors': len(errors)
        }), 200
        
    except Exception as e:
        logging.error(f"Error in batch predict: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred during batch prediction'}), 500

@api_bp.route('/model/info', methods=['GET'])
def model_info():
    """
    Get information about the trained model
    """
    try:
        predict_pipeline = PredictPipeline()
        
        model_info = {
            'model_type': type(predict_pipeline.model).__name__,
            'preprocessor_type': type(predict_pipeline.preprocessor).__name__,
            'features': [
                'Age', 'Gender', 'Tenure', 'Usage_Frequency', 'Support_Calls',
                'Payment_Delay', 'Subscription_Type', 'Contract_Length', 
                'Total_Spend', 'Last_Interaction'
            ],
            'feature_types': {
                'categorical': ['Gender', 'Contract_Length', 'Subscription_Type'],
                'numerical': ['Age', 'Tenure', 'Usage_Frequency', 'Support_Calls', 
                             'Payment_Delay', 'Total_Spend', 'Last_Interaction']
            },
            'api_version': '1.0.0'
        }
        
        return jsonify(model_info), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/model/features', methods=['GET'])
def get_feature_info():
    """
    Get detailed information about model features
    """
    feature_info = {
        'Age': {
            'type': 'numerical',
            'description': 'Customer age in years',
            'range': '18-100',
            'required': True
        },
        'Gender': {
            'type': 'categorical',
            'description': 'Customer gender',
            'values': ['Male', 'Female'],
            'required': True
        },
        'Tenure': {
            'type': 'numerical',
            'description': 'Number of months as customer',
            'range': '0-120',
            'required': True
        },
        'Usage_Frequency': {
            'type': 'numerical',
            'description': 'How often customer uses the service per month',
            'range': '0-50',
            'required': True
        },
        'Support_Calls': {
            'type': 'numerical',
            'description': 'Number of support calls made',
            'range': '0-50',
            'required': True
        },
        'Payment_Delay': {
            'type': 'numerical',
            'description': 'Average days of payment delay',
            'range': '0-90',
            'required': True
        },
        'Subscription_Type': {
            'type': 'categorical',
            'description': 'Type of subscription',
            'values': ['Basic', 'Standard', 'Premium'],
            'required': True
        },
        'Contract_Length': {
            'type': 'categorical',
            'description': 'Length of contract',
            'values': ['Monthly', 'Quarterly', 'Annual'],
            'required': True
        },
        'Total_Spend': {
            'type': 'numerical',
            'description': 'Total amount spent by customer',
            'range': '0-10000',
            'required': True
        },
        'Last_Interaction': {
            'type': 'numerical',
            'description': 'Days since last customer interaction',
            'range': '0-365',
            'required': True
        }
    }
    
    return jsonify(feature_info), 200

def validate_input_data(data):
    """
    Validate input data for type and range constraints
    """
    errors = []
    
    # Age validation
    try:
        age = float(data['Age'])
        if not (18 <= age <= 100):
            errors.append('Age must be between 18 and 100')
    except (ValueError, TypeError):
        errors.append('Age must be a valid number')
    
    # Gender validation
    if data['Gender'] not in ['Male', 'Female']:
        errors.append('Gender must be either Male or Female')
    
    # Tenure validation
    try:
        tenure = float(data['Tenure'])
        if not (0 <= tenure <= 120):
            errors.append('Tenure must be between 0 and 120 months')
    except (ValueError, TypeError):
        errors.append('Tenure must be a valid number')
    
    # Usage Frequency validation
    try:
        usage = float(data['Usage_Frequency'])
        if not (0 <= usage <= 50):
            errors.append('Usage Frequency must be between 0 and 50')
    except (ValueError, TypeError):
        errors.append('Usage Frequency must be a valid number')
    
    # Support Calls validation
    try:
        support = float(data['Support_Calls'])
        if not (0 <= support <= 50):
            errors.append('Support Calls must be between 0 and 50')
    except (ValueError, TypeError):
        errors.append('Support Calls must be a valid number')
    
    # Payment Delay validation
    try:
        delay = float(data['Payment_Delay'])
        if not (0 <= delay <= 90):
            errors.append('Payment Delay must be between 0 and 90 days')
    except (ValueError, TypeError):
        errors.append('Payment Delay must be a valid number')
    
    # Subscription Type validation
    if data['Subscription_Type'] not in ['Basic', 'Standard', 'Premium']:
        errors.append('Subscription Type must be Basic, Standard, or Premium')
    
    # Contract Length validation
    if data['Contract_Length'] not in ['Monthly', 'Quarterly', 'Annual']:
        errors.append('Contract Length must be Monthly, Quarterly, or Annual')
    
    # Total Spend validation
    try:
        spend = float(data['Total_Spend'])
        if not (0 <= spend <= 10000):
            errors.append('Total Spend must be between 0 and 10000')
    except (ValueError, TypeError):
        errors.append('Total Spend must be a valid number')
    
    # Last Interaction validation
    try:
        interaction = float(data['Last_Interaction'])
        if not (0 <= interaction <= 365):
            errors.append('Last Interaction must be between 0 and 365 days')
    except (ValueError, TypeError):
        errors.append('Last Interaction must be a valid number')
    
    return '; '.join(errors) if errors else None

def get_confidence_level(probability):
    """
    Get confidence level based on probability
    """
    if probability >= 0.8 or probability <= 0.2:
        return 'Very High'
    elif probability >= 0.7 or probability <= 0.3:
        return 'High'
    elif probability >= 0.6 or probability <= 0.4:
        return 'Medium'
    else:
        return 'Low'

def get_recommendation(prediction, probability):
    """
    Get business recommendation based on prediction and probability
    """
    if prediction == 1:
        if probability > 0.8:
            return "Immediate intervention required. Consider retention offers and personal outreach."
        elif probability > 0.6:
            return "High churn risk. Proactive retention efforts recommended."
        else:
            return "Monitor closely. Consider engagement initiatives."
    else:
        if probability < 0.3:
            return "Continue current engagement strategy. Consider upselling opportunities."
        elif probability < 0.5:
            return "Maintain current service levels. Customer shows good engagement."
        else:
            return "Moderate engagement. Consider additional value-added services."
