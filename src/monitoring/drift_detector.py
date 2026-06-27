import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import ColumnDriftMetric
import warnings
warnings.filterwarnings('ignore')

class DriftDetector:
    def __init__(self, reference_data_path="datasets/customer_churn_dataset-training-master.csv"):
        self.reference_data_path = reference_data_path
        self.report_dir = "artifacts/drift_reports"
        
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)

    def load_data(self):
        if not os.path.exists(self.reference_data_path):
            raise FileNotFoundError(f"Reference data not found at {self.reference_data_path}")
            
        reference = pd.read_csv(self.reference_data_path)
        
        from src.monitoring.database import engine
        current = pd.read_sql("SELECT * FROM prediction_logs", con=engine)
        
        if current.empty:
            raise ValueError("Prediction logs database is empty. Please make some predictions first.")
        
        # Ensure column names match where possible
        # Map prediction logs column names to reference column names if necessary
        return reference, current

    def generate_report(self):
        try:
            reference, current = self.load_data()
            
            # Create a report for data drift
            report = Report(metrics=[
                DataDriftPreset(),
            ])
            
            report.run(reference_data=reference, current_data=current)
            
            report_path = os.path.join(self.report_dir, "data_drift_report.html")
            report.save_html(report_path)
            print(f"Drift report generated successfully at {report_path}")
            
            # Extract drift status
            report_dict = report.as_dict()
            drift_detected = report_dict['metrics'][0]['result']['dataset_drift']
            print(f"Dataset drift detected: {drift_detected}")
            
            # Write to GITHUB_OUTPUT if running in GitHub Actions
            if "GITHUB_OUTPUT" in os.environ:
                with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                    f.write(f"drift_detected={str(drift_detected).lower()}\n")
            
            return report_path, drift_detected
            
        except Exception as e:
            print(f"Error generating drift report: {e}")
            return None, False

if __name__ == "__main__":
    detector = DriftDetector()
    detector.generate_report()
