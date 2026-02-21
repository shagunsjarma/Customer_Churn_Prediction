from flask import Flask, request, jsonify, render_template
import os
import sys
import pandas as pd
import numpy as np
from src.models.predict_pipeline import PredictPipeline, CustomData
from src.utils.exception import CustomException
from src.utils.logger import logging
import traceback

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/', methods=['GET'])
def home():
    """
    Home route to serve the main HTML template
    """
    try:
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error in home route: {str(e)}")
        return jsonify({'error': 'Failed to load the application'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint that takes customer data and returns churn prediction
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
                'error': f'Missing required fields: {", ".join(missing_fields)}'
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
            # Try to get prediction probabilities
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
            'message': get_churn_message(prediction[0], probability)
        }
        
        logging.info(f"Prediction made: {result}")
        return jsonify(result)
        
    except CustomException as e:
        logging.error(f"Custom exception in predict route: {str(e)}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logging.error(f"Unexpected error in predict route: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'An unexpected error occurred during prediction'}), 500

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

def get_churn_message(prediction, probability):
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

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    try:
        # Check if model files exist
        model_path = os.path.join('artifacts', 'model.pkl')
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        
        model_exists = os.path.exists(model_path)
        preprocessor_exists = os.path.exists(preprocessor_path)
        
        if model_exists and preprocessor_exists:
            return jsonify({
                'status': 'healthy',
                'model_loaded': True,
                'preprocessor_loaded': True
            }), 200
        else:
            return jsonify({
                'status': 'unhealthy',
                'model_loaded': model_exists,
                'preprocessor_loaded': preprocessor_exists,
                'message': 'Model or preprocessor files not found'
            }), 503
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """
    Get information about the trained model
    """
    try:
        # Check if user wants to use MLflow model
        use_mlflow = request.args.get('use_mlflow', 'false').lower() == 'true'
        model_version = request.args.get('version', None)
        
        if use_mlflow:
            predict_pipeline = PredictPipeline(use_mlflow_model=True, model_version=model_version)
        else:
            predict_pipeline = PredictPipeline()
        
        # Try to get model information
        model_info = {
            'model_type': type(predict_pipeline.model).__name__,
            'preprocessor_type': type(predict_pipeline.preprocessor).__name__,
            'model_source': 'MLflow' if use_mlflow else 'Local',
            'model_version': model_version if use_mlflow else 'N/A',
            'features': [
                'Age', 'Gender', 'Tenure', 'Usage_Frequency', 'Support_Calls',
                'Payment_Delay', 'Subscription_Type', 'Contract_Length', 
                'Total_Spend', 'Last_Interaction'
            ]
        }
        
        return jsonify(model_info), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_mlflow', methods=['POST'])
def predict_mlflow():
    """
    Prediction endpoint using MLflow model
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
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Validate data types and ranges
        validation_errors = validate_input_data(data)
        if validation_errors:
            return jsonify({'error': validation_errors}), 400
        
        # Get model version from request
        model_version = data.get('model_version', None)
        
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
        
        # Initialize prediction pipeline with MLflow model
        predict_pipeline = PredictPipeline(use_mlflow_model=True, model_version=model_version)
        
        # Make prediction with probability
        prediction, probabilities = predict_pipeline.predict_with_probability(df)
        
        # Get prediction probability
        if probabilities is not None:
            probability = float(probabilities[0][1])  # Probability of churn (class 1)
        else:
            probability = 0.8 if prediction[0] == 1 else 0.2
        
        # Prepare response
        result = {
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'churn_risk': 'High' if prediction[0] == 1 else 'Low',
            'message': get_churn_message(prediction[0], probability),
            'model_source': 'MLflow',
            'model_version': model_version or 'latest'
        }
        
        logging.info(f"MLflow prediction made: {result}")
        return jsonify(result)
        
    except CustomException as e:
        logging.error(f"Custom exception in MLflow predict route: {str(e)}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logging.error(f"Unexpected error in MLflow predict route: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'An unexpected error occurred during prediction'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Check if artifacts directory exists
    artifacts_dir = 'artifacts'
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)
        logging.warning(f"Created artifacts directory: {artifacts_dir}")
    
    # Check if model files exist
    model_path = os.path.join(artifacts_dir, 'model.pkl')
    preprocessor_path = os.path.join(artifacts_dir, 'preprocessor.pkl')
    
    if not os.path.exists(model_path):
        logging.warning(f"Model file not found: {model_path}")
        logging.warning("Please train the model first using the training pipeline")
    
    if not os.path.exists(preprocessor_path):
        logging.warning(f"Preprocessor file not found: {preprocessor_path}")
        logging.warning("Please run the data transformation pipeline first")
    
    # Run the Flask app
    logging.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
