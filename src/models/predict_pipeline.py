import os
import sys
import pandas as pd
from src.utils.save_model import load_object
from src.utils.exception import CustomException

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        
    def predict(self, features):
        try:
            if isinstance(features, dict):
                df = pd.DataFrame([features])
            elif isinstance(features, pd.DataFrame):
                df = features.copy()
            else:
                raise CustomException("Input features should be a dictionary or a pandas DataFrame", sys)
            
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)
            data_transformed = preprocessor.transform(df)
            preds = model.predict(data_transformed)
            return preds
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self, 
                 Age: float,
                 Gender: str,
                 Tenure: float,
                 Usage_Frequency: float,
                 Support_Calls: float,
                 Payment_Delay: float,
                 Subscription_Type: str,
                 Contract_Length: str,
                 Total_Spend: float,
                 Last_Interaction: float
                 ):  
        self.Age = Age
        self.Gender = Gender
        self.Tenure = Tenure
        self.Usage_Frequency = Usage_Frequency
        self.Support_Calls = Support_Calls
        self.Payment_Delay = Payment_Delay
        self.Subscription_Type = Subscription_Type
        self.Contract_Length = Contract_Length
        self.Total_Spend = Total_Spend
        self.Last_Interaction = Last_Interaction
        
    def get_data_as_dataframe(self):
        try:
            data = {
                "Age": [self.Age],
                "Gender": [self.Gender],
                "Tenure": [self.Tenure],
                "Usage Frequency": [self.Usage_Frequency],
                "Support Calls": [self.Support_Calls],
                "Payment Delay": [self.Payment_Delay],
                "Subscription Type": [self.Subscription_Type],
                "Contract Length": [self.Contract_Length],
                "Total Spend": [self.Total_Spend],
                "Last Interaction": [self.Last_Interaction]
            }
            df = pd.DataFrame(data) 
            return df
        except Exception as e:
            raise CustomException(e, sys)              