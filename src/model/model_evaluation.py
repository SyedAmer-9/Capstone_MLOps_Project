import numpy as np
import pandas as pd
import os
import pickle
import json
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
import yaml
import sys
import mlflow 
import mlflow.sklearn
import dagshub
from scipy.sparse import load_npz
from src.logger import logging
from dotenv import load_dotenv

# -----------------------------------------------------------------
# THE FIX: Robustly find and load the .env file
# -----------------------------------------------------------------
# Get the directory of this script (src/model)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Walk up to the project root (Capstone_MLOps_Project)
# src/model -> src -> Capstone_MLOps_Project
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))

# Define the absolute path to the .env file
ENV_FILE_PATH = os.path.join(PROJECT_ROOT, ".env")

# Load it explicitly
if os.path.exists(ENV_FILE_PATH):
    load_dotenv(dotenv_path=ENV_FILE_PATH)
    logging.info(f"Loaded environment variables from: {ENV_FILE_PATH}")
else:
    logging.warning(f"WARNING: .env file not found at {ENV_FILE_PATH}")

# Verify the token is loaded
dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
if not dagshub_token:
    raise EnvironmentError(f"DAGSHUB_USER_TOKEN is missing. Checked file: {ENV_FILE_PATH}")

load_dotenv()

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug(f'Parameters retrieved from {params_path}')
        return params
    except FileNotFoundError:
        logging.error(f'File not found: {params_path}')
        raise
    except Exception as e:
        logging.error(f'Unexpected error: {e}')
        raise

def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info(f'Model loaded from {file_path}')
        return model
    except Exception as e:
        logging.error(f'Unexpected error occurred while loading the model: {e}')
        raise

def load_processed_data(processed_data_path: str, test_file_name: str) -> tuple:
    try:
        logging.info(f"Loading processed test data from {processed_data_path}")
        X_test = load_npz(os.path.join(processed_data_path, "X_test.npz"))
        y_test = np.load(os.path.join(processed_data_path, "y_test.npy"))
        
        logging.info("Processed test data loaded successfully.")
        return X_test, y_test
    except Exception as e:
        logging.error(f'Error loading processed data: {e}')
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logging.info(f'Model evaluation metrics calculated: {metrics_dict}')
        return metrics_dict
    except Exception as e:
        logging.error(f'Error during model evaluation: {e}')
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            info_abs_path = os.path.join(project_root, file_path)
            
            model_info = {'run_id': run_id, 'model_path': model_path}
            os.makedirs(os.path.dirname(info_abs_path), exist_ok=True)
            with open(info_abs_path, 'w') as file:
                json.dump(model_info, file, indent=4)
            logging.debug(f'Model info saved to {info_abs_path}')
        except Exception as e:
            logging.error(f'Error occurred while saving the model info: {e}')
            raise

def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        # Ensure the 'reports' directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info(f'Metrics saved to {file_path}')
    except Exception as e:
        logging.error(f'Error occurred while saving the metrics: {e}')
        raise

def setup_mlflow(params:dict):
    logging.info("Setting up MLFLOW...")

    dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
    #dagshub_token = "bbb0e8511a3a90b1fb5e9b9fd55b8b695bcf3113"
    #print(f"DEBUG: Token found in .env: {dagshub_token} ")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB USER TOKEN env variable is not set")
    os.environ["MLFLOW_TRACKING_USERNAME"]=dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"]=dagshub_token

    mlflow.set_tracking_uri(params['mlflow_tracking_uri'])
    mlflow.set_experiment(params['experiment_name'])
    logging.info(f"MLFlow experiment set to :{params['experiment_name']}")

def main():
    try:
        params = load_params('params.yaml')

        model_path = os.path.join(params['model_training']['model_save_path'],
                                  params['model_training']['model_file_name'])        
        processed_data_path = params['feature_engineering']['processed_data_path']
        test_file_name = params['feature_engineering']['test_data_file']

        metrics_path = params['model_evaluation']['metrics_path']
        experiment_info_path = params['model_evaluation']['experiment_info_path']

        setup_mlflow(params['mlflow_config'])

        with mlflow.start_run(run_name="Final Model Evaluation") as run:
            clf = load_model(file_path=model_path)
            X_test,y_test = load_processed_data(processed_data_path,test_file_name)

            metrics = evaluate_model(clf,X_test,y_test)

            save_metrics(metrics,file_path=metrics_path)

            logging.info("Logging parameters and metrics to MLflow...")
            mlflow.log_params(params['model_training']['LogisticRegression']['params'])
            mlflow.log_metrics(metrics)

            logging.info("Logging model and metrics artifact to MLflow...")
            mlflow.sklearn.log_model(clf, "model")
            mlflow.log_artifact(metrics_path)

            save_model_info(run.info.run_id, "model", experiment_info_path)
            logging.info(f"Run ID info saved to {experiment_info_path}")

            logging.info("Model evaluation and logging complete.")

    except Exception as e:
        logging.error('Failed to complete the model evaluation process')
        logging.exception(e) # Log the full traceback
        raise e

if __name__ == '__main__':
    main()