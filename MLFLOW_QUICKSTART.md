# MLflow Quick Start Guide

Get up and running with MLflow integration in 5 minutes!

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
python setup_mlflow.py install
```

### 2. Complete Setup
```bash
python setup_mlflow.py all
```

### 3. Start MLflow UI
```bash
python setup_mlflow.py ui
```
Open http://localhost:5000 in your browser

## 📊 What You Get

- **Experiment Tracking**: All model training runs are automatically logged
- **Model Registry**: Best models are automatically registered and versioned
- **Model Serving**: Use MLflow models in your Flask API
- **Web UI**: Visual interface to compare models and track experiments

## 🧪 Test Everything Works

```bash
python test_mlflow_integration.py
```

## 🔄 Training with MLflow

```bash
# Train models (automatically logs to MLflow)
python src/models/trainer_pipeline.py
```

## 🔮 Making Predictions

### Local Model (Default)
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

### MLflow Model
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

## 📋 Management Commands

```bash
# List experiments
python mlflow_manage.py list-experiments

# List models
python mlflow_manage.py list-models

# Get model info
python mlflow_manage.py model-info

# Export model
python mlflow_manage.py export-model
```

## 🎯 Key Features

### Automatic Logging
- ✅ Model parameters and hyperparameters
- ✅ Training metrics (accuracy, precision, recall, F1, ROC-AUC)
- ✅ Dataset information
- ✅ Model artifacts

### Model Comparison
- ✅ Compare multiple algorithms automatically
- ✅ Track best performing model
- ✅ Version control for models

### Production Ready
- ✅ Load models from MLflow registry
- ✅ Model versioning and staging
- ✅ Integration with Flask API

## 🔧 Troubleshooting

### Port Already in Use
```bash
python mlflow_manage.py start-ui --port 5001
```

### Model Not Found
```bash
# Check if models exist
python mlflow_manage.py list-models

# Train models first
python setup_mlflow.py train
```

### Dependencies Issues
```bash
pip install -r requirements.txt
```

## 📚 Next Steps

1. **Explore MLflow UI**: http://localhost:5000
2. **Read Full Documentation**: [MLFLOW_INTEGRATION.md](MLFLOW_INTEGRATION.md)
3. **Experiment**: Try different hyperparameters and see results
4. **Deploy**: Use exported models for production deployment

## 🎉 You're All Set!

Your churn prediction project now has full MLflow integration with experiment tracking, model registry, and serving capabilities!
