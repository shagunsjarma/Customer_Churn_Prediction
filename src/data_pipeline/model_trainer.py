import os
import sys
from dataclasses import dataclass
from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.save_model import save_object, load_object
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGB Classifier": XGBClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier()
            }
            model_report = {}
            for i in range(len(models)):
                model = list(models.values())[i]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                model_report[list(models.keys())[i]] = acc
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with accuracy: {model_report[best_model_name]}")
            if model_report[best_model_name] < 0.5:
                logging.warning(f"Best model accuracy {model_report[best_model_name]} is below 50%, but continuing with training")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)
            final_score = accuracy_score(y_test, predicted)
            return final_score
        except Exception as e:
            raise CustomException(e, sys)