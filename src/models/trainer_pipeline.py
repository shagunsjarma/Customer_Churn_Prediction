from src.data_pipeline.data_ingestion import DataIngestion
from src.data_pipeline.data_transformer import DataTransformation
from src.data_pipeline.model_trainer import ModelTrainer
from src.utils.logger import logging
import os
import sys

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
    