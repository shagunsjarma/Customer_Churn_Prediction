# 🎯 Customer Churn Prediction

A production-ready machine learning project that predicts customer churn risk using advanced algorithms and provides actionable business insights for customer retention strategies.

## 📊 Project Overview

This project implements a complete end-to-end machine learning pipeline for customer churn prediction. It is designed with production readiness in mind, featuring:
- **Machine Learning Pipeline**: Complete ML workflow from data ingestion to model deployment using Scikit-Learn and XGBoost.
- **REST API & Web Interface**: A robust Flask application serving a responsive HTML interface and a comprehensive REST API.
- **Production Server**: Configured with Gunicorn for reliable, concurrent request handling.
- **Environment Management**: Secure configuration using `python-dotenv`.
- **Model Tracking**: Deep integration with MLflow for experiment tracking and model versioning.
- **Automated CI/CD**: GitHub Actions pipeline that automatically trains the model, tests the API, and builds the Docker image.

## 🏗️ Project Structure

```
churn_prediction/
├── app.py                          # Main Flask application
├── requirements.txt                # Dependencies (includes Gunicorn, Dotenv)
├── Dockerfile                      # Production Docker configuration
├── .env.example                    # Environment configuration template
├── templates/                      # Web interface (index.html)
├── src/                            # Core application logic
│   ├── api/                        # API routes, middleware, and validators
│   ├── data_pipeline/              # Data ingestion and preprocessing
│   ├── models/                     # Model training and prediction pipelines
│   └── utils/                      # Utilities (exceptions, logger, MLflow config)
├── datasets/                       # Raw and processed data files
├── artifacts/                      # Serialized models (model.pkl, preprocessor.pkl)
└── logs/                           # Auto-generated application logs
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- Git

### Local Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd churn_prediction
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv cenv
   # On Windows:
   cenv\Scripts\activate
   # On macOS/Linux:
   source cenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   Create a `.env` file in the root directory:
   ```bash
   cp .env.example .env
   ```
   *Edit `.env` to match your local setup.*

## 🚀 Quick Start

### 1. Train the Model
Generate the required `model.pkl` and `preprocessor.pkl` artifacts:
```bash
# Set Python path and run training
$env:PYTHONPATH="." ; python src/models/trainer_pipeline.py
# Or on Linux/macOS: PYTHONPATH=. python src/models/trainer_pipeline.py
```

### 2. Start the Application
Run the Flask server (locally):
```bash
python app.py
```
Visit `http://localhost:5000` to access the web interface.

*For production, use Gunicorn (as defined in the Dockerfile):*
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 3. Docker Deployment
Build and run the containerized application:
```bash
docker build -t churn-prediction-app:latest .
docker run -p 5000:5000 --env-file .env churn-prediction-app:latest
```

## 🌐 API Reference

**Base URL**: `http://localhost:5000/api/v1`

### 1. Health Check
```bash
curl -X GET http://localhost:5000/api/v1/health
```

### 2. Single Prediction
```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
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
```

### 3. Batch Prediction
```bash
curl -X POST http://localhost:5000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "customer_id": "CUST001",
        "Age": 35,
        "Gender": "Female",
        ...
      }
    ]
  }'
```

### 4. MLflow Prediction
Predict using a specific model version logged in MLflow:
```bash
curl -X POST http://localhost:5000/api/v1/predict_mlflow \
  -H "Content-Type: application/json" \
  -d '{
    "model_version": "1",
    "Age": 35,
    ...
  }'
```

## 🧪 Testing

Run the full integration test suite:
```bash
PYTHONPATH=. python test_api.py
```

## ⚙️ Production Readiness
This application has been upgraded for enterprise deployment:
- **Gunicorn** WSGI server used for robust connection handling.
- **Environment Management** via `dotenv` ensures sensitive configurations remain out of the codebase.
- **Error Sanitization**: API routes mask internal server errors (HTTP 500) from the client while preserving detailed tracebacks in the internal `logs/` directory.
- **Automated CI/CD**: GitHub Actions workflow (`ci_cd.yml`) automatically provisions the environment, trains the model, and validates endpoints on every pull request.

---
**Built with ❤️ for customer retention and business intelligence**
