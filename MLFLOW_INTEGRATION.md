# MLflow Integration for Churn Prediction Project

This document describes the MLflow integration for the churn prediction project, including setup, usage, and management.

## Overview

MLflow has been integrated into the churn prediction project to provide:
- **Experiment Tracking**: Track different model training runs and compare performance
- **Model Registry**: Store and version trained models
- **Model Serving**: Serve models directly from MLflow registry
- **Reproducibility**: Track parameters, metrics, and artifacts for each experiment

## Features

### 1. Experiment Tracking
- Automatic logging of model parameters and hyperparameters
- Comprehensive metrics tracking (accuracy, precision, recall, F1-score, ROC-AUC)
- Dataset information logging
- Model comparison across different algorithms

### 2. Model Registry
- Automatic model registration after training
- Version control for models
- Model metadata tracking
- Easy model retrieval for serving

### 3. Model Serving
- Load models from MLflow registry for predictions
- Support for different model versions
- Integration with Flask API for real-time predictions

## Setup

### 1. Install Dependencies

```bash
# Install all required packages
python setup_mlflow.py install

# Or install manually
pip install -r requirements.txt
```

### 2. Complete Setup

```bash
# Run complete setup (install, configure, train, and verify)
python setup_mlflow.py all
```

### 3. Individual Setup Steps

```bash
# Set up MLflow tracking
python setup_mlflow.py setup

# Train and log model to MLflow
python setup_mlflow.py train

# Verify setup
python setup_mlflow.py verify

# Start MLflow UI
python setup_mlflow.py ui
```

## Usage

### 1. Training with MLflow

The training pipeline automatically logs to MLflow:

```python
# Run the training pipeline
python src/models/trainer_pipeline.py
```

This will:
- Create a new MLflow run
- Log all model parameters and hyperparameters
- Track metrics for each model (Random Forest, Decision Tree, etc.)
- Register the best performing model
- Save model artifacts

### 2. Making Predictions with MLflow Models

#### Using Flask API

**Local Model (Default):**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 45,
    "Gender": "Male",
    "Tenure": 24,
    "Usage_Frequency": 15,
    "Support_Calls": 2,
    "Payment_Delay": 5,
    "Subscription_Type": "Premium",
    "Contract_Length": "Annual",
    "Total_Spend": 1200,
    "Last_Interaction": 30
  }'
```

**MLflow Model:**
```bash
curl -X POST http://localhost:5000/predict_mlflow \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 45,
    "Gender": "Male",
    "Tenure": 24,
    "Usage_Frequency": 15,
    "Support_Calls": 2,
    "Payment_Delay": 5,
    "Subscription_Type": "Premium",
    "Contract_Length": "Annual",
    "Total_Spend": 1200,
    "Last_Interaction": 30,
    "model_version": "1"
  }'
```

#### Using Python Code

```python
from src.models.predict_pipeline import PredictPipeline

# Load model from MLflow
predictor = PredictPipeline(use_mlflow_model=True, model_version="1")

# Make prediction
features = {
    "Age": 45,
    "Gender": "Male",
    "Tenure": 24,
    # ... other features
}

prediction = predictor.predict(features)
print(f"Prediction: {prediction}")
```

### 3. MLflow Management

Use the management script to interact with MLflow:

```bash
# List all experiments
python mlflow_manage.py list-experiments

# List runs for an experiment
python mlflow_manage.py list-runs --experiment churn_prediction

# List registered models
python mlflow_manage.py list-models

# Get model information
python mlflow_manage.py model-info --model churn_prediction_model

# Load model for testing
python mlflow_manage.py load-model --model churn_prediction_model --version 1

# Export model for deployment
python mlflow_manage.py export-model --model churn_prediction_model --version 1

# Start MLflow UI
python mlflow_manage.py start-ui --port 5001
```

## MLflow UI

Access the MLflow UI to:
- View experiments and runs
- Compare model performance
- Download model artifacts
- Manage model registry

```bash
# Start UI
python setup_mlflow.py ui
# or
python mlflow_manage.py start-ui

# Access at: http://localhost:5000
```

## Project Structure

```
├── src/
│   ├── utils/
│   │   └── mlflow_config.py          # MLflow configuration and utilities
│   ├── data_pipeline/
│   │   └── model_trainer.py          # Enhanced with MLflow integration
│   └── models/
│       └── predict_pipeline.py       # Enhanced with MLflow model loading
├── mlflow_manage.py                  # MLflow management script
├── setup_mlflow.py                   # Setup and configuration script
├── requirements.txt                  # Updated with MLflow dependencies
└── MLFLOW_INTEGRATION.md             # This documentation
```

## Configuration

### Environment Variables

```bash
# Set MLflow tracking URI (optional)
export MLFLOW_TRACKING_URI="file:///path/to/mlruns"

# For remote tracking server
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

### MLflow Configuration

The MLflow configuration is managed in `src/utils/mlflow_config.py`:

- **Experiment Name**: `churn_prediction`
- **Model Name**: `churn_prediction_model`
- **Tracking URI**: Configurable via environment variable

## Monitoring and Logging

### Training Metrics

The following metrics are automatically logged:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- ROC-AUC (when available)

### Parameters Logged

- Model hyperparameters
- Dataset information
- Training configuration
- Feature names and types

### Model Artifacts

- Trained model files
- Preprocessing pipeline
- Model metadata
- Training logs

## Best Practices

### 1. Experiment Organization
- Use descriptive run names
- Tag important runs
- Document experimental goals

### 2. Model Versioning
- Use semantic versioning for models
- Tag production-ready models
- Maintain model lineage

### 3. Performance Monitoring
- Track prediction metrics
- Monitor model drift
- Set up alerts for performance degradation

### 4. Reproducibility
- Use fixed random seeds
- Log all dependencies
- Save data preprocessing steps

## Troubleshooting

### Common Issues

1. **MLflow UI not starting**
   ```bash
   # Check if port is already in use
   netstat -an | grep 5000
   
   # Use different port
   python mlflow_manage.py start-ui --port 5001
   ```

2. **Model loading errors**
   ```bash
   # Verify model exists
   python mlflow_manage.py list-models
   
   # Check model versions
   python mlflow_manage.py model-info
   ```

3. **Permission errors**
   ```bash
   # Ensure write permissions for mlruns directory
   chmod -R 755 mlruns/
   ```

### Logs and Debugging

Check the application logs for detailed error information:
```bash
# View training logs
tail -f logs/training.log

# View prediction logs
tail -f logs/prediction.log
```

## Advanced Usage

### Custom Model Registry

For production environments, consider using a remote MLflow server:

```bash
# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0

# Set tracking URI
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

### Model Deployment

For production deployment, you can:

1. **Export models**:
   ```bash
   python mlflow_manage.py export-model --model churn_prediction_model
   ```

2. **Use MLflow serving**:
   ```bash
   mlflow models serve -m "models:/churn_prediction_model/1" --port 1234
   ```

3. **Deploy to cloud platforms** using MLflow's deployment tools.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review MLflow documentation: https://mlflow.org/docs/
3. Check project logs for detailed error messages
4. Verify all dependencies are installed correctly
