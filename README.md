# 🎯 Customer Churn Prediction

A machine learning project that predicts customer churn risk using advanced algorithms and provides actionable business insights for customer retention strategies.

## 📊 Project Overview

This project implements a complete end-to-end machine learning pipeline for customer churn prediction, including data preprocessing, model training, evaluation, and deployment through both web interface and REST API.

## 🚀 Features

- **Machine Learning Pipeline**: Complete ML workflow from data ingestion to model deployment
- **Multiple Algorithms**: Random Forest, XGBoost, Decision Tree, Logistic Regression, AdaBoost
- **Web Interface**: Beautiful, responsive HTML interface for predictions
- **REST API**: Comprehensive API with validation, rate limiting, and documentation
- **Batch Processing**: Support for single and batch predictions
- **Business Intelligence**: Confidence levels and actionable recommendations
- **Model Persistence**: Save and load trained models and preprocessors
- **Comprehensive Logging**: Detailed logging for monitoring and debugging

## 🏗️ Project Structure

```
churn_prediction/
├── app.py                          # Main Flask application
├── templates/
│   └── index.html                  # Web interface
├── src/
│   ├── api/                        # API package
│   │   ├── routes.py              # API endpoints
│   │   ├── middleware.py          # Request middleware
│   │   ├── validators.py          # Data validation
│   │   └── documentation.py       # API documentation
│   ├── data_pipeline/             # Data processing
│   │   ├── data_ingestion.py     # Data loading and splitting
│   │   ├── data_transformer.py   # Feature preprocessing
│   │   └── model_trainer.py      # Model training
│   ├── models/                    # Model management
│   │   ├── predict_pipeline.py   # Prediction pipeline
│   │   └── trainer_pipeline.py   # Training pipeline
│   └── utils/                     # Utilities
│       ├── exception.py          # Custom exceptions
│       ├── logger.py             # Logging configuration
│       └── save_model.py         # Model persistence
├── datasets/                      # Data files
├── artifacts/                     # Trained models and preprocessors
├── logs/                         # Log files
├── notebooks/                    # Jupyter notebooks
├── test_api.py                   # API testing script
└── requirements.txt              # Dependencies
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd churn_prediction
   ```

2. **Create virtual environment**
   ```bash
   # Using venv
   python -m venv cenv
   source cenv/bin/activate  # On Windows: cenv\Scripts\activate

   # Or using conda
   conda create -n churn_prediction python=3.8
   conda activate churn_prediction
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data**
   - Place your dataset in the `datasets/` directory
   - Ensure the dataset has the required columns (see Data Format section)

## 🚀 Quick Start

### 1. Train the Model

```bash
# Set Python path and run training
$env:PYTHONPATH = "."; python src/models/trainer_pipeline.py
```

### 2. Start the Web Application

```bash
python app.py
```

Visit `http://localhost:5000` to access the web interface.

### 3. Test the API

```bash
python test_api.py
```

## 📊 Data Format

The model expects the following features:

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| Age | Numerical | 18-100 | Customer age in years |
| Gender | Categorical | Male/Female | Customer gender |
| Tenure | Numerical | 0-120 | Months as customer |
| Usage_Frequency | Numerical | 0-50 | Service usage per month |
| Support_Calls | Numerical | 0-50 | Number of support calls |
| Payment_Delay | Numerical | 0-90 | Average payment delay in days |
| Subscription_Type | Categorical | Basic/Standard/Premium | Service tier |
| Contract_Length | Categorical | Monthly/Quarterly/Annual | Contract duration |
| Total_Spend | Numerical | 0-10000 | Total customer spending |
| Last_Interaction | Numerical | 0-365 | Days since last interaction |
| Churn | Target | 0/1 | Churn indicator (0=Stay, 1=Churn) |

## 🌐 API Usage

### Base URL
```
http://localhost:5000/api/v1
```

### Endpoints

#### Health Check
```bash
curl -X GET http://localhost:5000/api/v1/health
```

#### Single Prediction
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

#### Batch Prediction
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

#### Model Information
```bash
curl -X GET http://localhost:5000/api/v1/model/info
```

#### API Documentation
```bash
curl -X GET http://localhost:5000/api/docs
```

## 📈 Model Performance

- **Algorithm**: Ensemble of Random Forest, XGBoost, Decision Tree, Logistic Regression, AdaBoost
- **Accuracy**: ~57% on test dataset
- **Features**: 10 customer attributes
- **Preprocessing**: StandardScaler for numerical features, OneHotEncoder for categorical
- **Validation**: Cross-validated on holdout test set

## 🎯 Business Applications

### Customer Retention
- Identify at-risk customers for targeted retention campaigns
- Prioritize retention efforts based on churn probability
- Develop personalized retention strategies

### Resource Allocation
- Focus customer success teams on high-risk accounts
- Optimize marketing spend on retention vs. acquisition
- Allocate support resources efficiently

### Strategic Planning
- Understand churn patterns and drivers
- Develop data-driven retention strategies
- Monitor customer health metrics

## 🔧 Configuration

### Environment Variables
- `PYTHONPATH`: Set to project root for module imports
- `FLASK_ENV`: Set to 'development' or 'production'

### Model Configuration
- Model files are stored in `artifacts/` directory
- Preprocessor is saved as `artifacts/preprocessor.pkl`
- Trained model is saved as `artifacts/model.pkl`

### Logging
- Logs are stored in `logs/` directory
- Log level can be configured in `src/utils/logger.py`
- API requests are logged with timing information

## 🧪 Testing

### Run API Tests
```bash
python test_api.py
```

### Test Individual Components
```bash
# Test data ingestion
python -c "from src.data_pipeline.data_ingestion import DataIngestion; DataIngestion().initiate_data_ingestion()"

# Test data transformation
python -c "from src.data_pipeline.data_transformer import DataTransformation; DataTransformation().get_data_transformer_object()"

# Test prediction
python -c "from src.models.predict_pipeline import PredictPipeline; print('Model loaded successfully')"
```

## 📚 Development

### Adding New Features
1. Create feature in appropriate module
2. Add validation in `src/api/validators.py`
3. Update API documentation in `src/api/documentation.py`
4. Add tests in `test_api.py`

### Model Improvements
1. Modify `src/data_pipeline/model_trainer.py` for new algorithms
2. Update feature engineering in `src/data_pipeline/data_transformer.py`
3. Retrain model using `src/models/trainer_pipeline.py`

### API Extensions
1. Add new endpoints in `src/api/routes.py`
2. Implement middleware in `src/api/middleware.py`
3. Update documentation in `src/api/documentation.py`

## 🚀 Deployment

### Local Deployment
1. Train the model: `python src/models/trainer_pipeline.py`
2. Start the application: `python app.py`
3. Access at `http://localhost:5000`

### Production Deployment
1. Set up production environment
2. Configure logging and monitoring
3. Use production WSGI server (e.g., Gunicorn)
4. Set up reverse proxy (e.g., Nginx)
5. Configure SSL certificates

### Docker Deployment
```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 📊 Monitoring

### Health Checks
- API health: `GET /api/v1/health`
- Model status: Check if model files exist and are loadable
- System metrics: Monitor CPU, memory, and disk usage

### Logging
- Application logs: `logs/` directory
- API request logs: Include timing and error information
- Model prediction logs: Track prediction requests and results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Commit your changes: `git commit -m 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues

**Model not loading:**
- Ensure model files exist in `artifacts/` directory
- Check file permissions
- Verify Python path is set correctly

**API errors:**
- Check input data format
- Verify all required fields are provided
- Check validation constraints

**Import errors:**
- Set `PYTHONPATH` to project root
- Ensure all dependencies are installed
- Check virtual environment activation

### Getting Help

- Check the logs in `logs/` directory
- Review API documentation at `/api/docs`
- Test individual components using provided scripts
- Open an issue for bugs or feature requests

## 📈 Future Enhancements

- [ ] Real-time model retraining
- [ ] A/B testing framework
- [ ] Advanced feature engineering
- [ ] Model explainability (SHAP, LIME)
- [ ] Integration with customer databases
- [ ] Automated model monitoring
- [ ] Multi-model ensemble voting
- [ ] Time-series churn prediction

## 🙏 Acknowledgments

- Scikit-learn for machine learning algorithms
- Flask for web framework
- Pandas for data manipulation
- XGBoost for gradient boosting
- Gradio for web interface components

---

**Built with ❤️ for customer retention and business intelligence**
