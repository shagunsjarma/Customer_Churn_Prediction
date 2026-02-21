import sys
from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.save_model import save_object
from src.utils.mlflow_config import MLflowConfig
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from dataclasses import dataclass
import os
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from datetime import datetime


class DataTransformationConfig:
    data_transformation_config_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.mlflow_config = MLflowConfig()
        
    def get_data_transformer_object(self):
        categorical_cols = ["Gender", "Contract Length", "Subscription Type"]
        numerical_cols = ["Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Total Spend", "Last Interaction"]
        
        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])
        preprocessor = ColumnTransformer([
            ('num_pipeline', num_pipeline, numerical_cols),
            ('cat_pipeline', cat_pipeline, categorical_cols)
        ])
        return preprocessor
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            processing = self.get_data_transformer_object()
            target_column_name = "Churn"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name].fillna(0)  # Fill NaN with 0
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name].fillna(0)  # Fill NaN with 0
            logging.info("Applying preprocessing object on training and testing dataframes")
            input_feature_train_arr = processing.fit_transform(input_feature_train_df)
            input_feature_test_arr = processing.transform(input_feature_test_df)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Saved preprocessing object")
            
            # Save preprocessor locally (existing functionality)
            save_object(
                file_path=self.data_transformation_config.data_transformation_config_file_path,
                obj=processing
            )
            
            # Log preprocessor to MLflow (optional - can be done in training pipeline)
            # Uncomment the following lines if you want to log preprocessor separately
            # with self.mlflow_config.start_run(run_name=f"preprocessor_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            #     self.mlflow_config.log_preprocessor(
            #         preprocessor=processing,
            #         preprocessor_name="churn_prediction_preprocessor"
            #     )
            #     # Log transformation parameters
            #     transform_params = {
            #         "categorical_columns": ["Gender", "Contract Length", "Subscription Type"],
            #         "numerical_columns": ["Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Total Spend", "Last Interaction"],
            #         "imputation_strategy_numerical": "median",
            #         "imputation_strategy_categorical": "most_frequent",
            #         "scaling_method": "StandardScaler",
            #         "encoding_method": "OneHotEncoder"
            #     }
            #     self.mlflow_config.log_model_parameters(transform_params)
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.data_transformation_config_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        

