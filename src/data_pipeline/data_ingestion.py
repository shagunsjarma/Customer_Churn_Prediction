import os
import sys
from src.utils.exception import CustomException
from src.utils.logger import logging
from src.data_pipeline.data_transformer import DataTransformation
from src.data_pipeline.model_trainer import ModelTrainer
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
import numpy as np


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("datasets/customer_churn_dataset.csv")
            logging.info("Read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")
            train_set = pd.read_csv("datasets/customer_churn_dataset-training-master.csv")
            test_set = pd.read_csv("datasets/customer_churn_dataset-testing-master.csv")
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) 
            logging.info("Ingestion of the data is completed")
            return (
                self.ingestion_config.raw_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
        
if __name__ == "__main__":
    obj = DataIngestion()
    raw_data_path, train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    print("Data transformation completed")
    model_trainer = ModelTrainer()
    print("Model training started")
    accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Accuracy of the model is {accuracy}")
    