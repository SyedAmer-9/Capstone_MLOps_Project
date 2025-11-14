import json
import mlflow
import logging
import os
import yaml
import sys
import dagshub

from src.logger import logging
from dotenv import load_dotenv

load_dotenv()

def load_params(params_path: str) -> dict:
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        params_file_abs_path = os.path.join(project_root, params_path)
        with open(params_file_abs_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug(f'Parameters retrieved from {params_file_abs_path}')
        return params
    except FileNotFoundError:
        logging.error(f'File not found: {params_file_abs_path}')
        raise
    except Exception as e:
        logging.error(f'Unexpected error: {e}')
        raise

def setup_mlflow(params:dict):
    logging.info("setting up the MLFlow for registration..")

    dagshub_token = os.getenv('DAGSHUB_USER_TOKEN')

    if not dagshub_token:
        logging.info("DAGSHUB USER TOKEN ERROR")
        raise EnvironmentError("DAGSHUB USER TOKEN ERROR")
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
    os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

    mlflow.set_tracking_uri(params['mlflow_tracking_uri'])
    mlflow.set_experiment(params['experiment_name'])
    logging.info(f"MLflow experiment set to: {params['experiment_name']}")

def load_model_info(file_path: str) -> dict:
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        info_abs_path = os.path.join(project_root, file_path)
        
        with open(info_abs_path, 'r') as file:
            model_info = json.load(file)
        logging.debug(f'Model info loaded from {info_abs_path}')
        return model_info
    except FileNotFoundError:
        logging.error(f'File not found: {info_abs_path}')
        raise
    except Exception as e:
        logging.error(f'Unexpected error occurred while loading the model info: {e}')
        raise

def register_model(model_name:str,model_info:dict):
    try:
        model_uri = f'runs:/{model_info['run_id']}/{model_info['model_path']}'
        logging.info(f"Registering model from URI :{model_uri}")

        model_version=mlflow.register_model(model_uri,model_name)

        client=mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage='Staging'
        )

        logging.info(f"Model {model_name} version {model_version.version} registered ans transitioned to staging")

    except Exception as e:
        logging.info("error during model registration : %s",e)
        raise
def main():
    try:
        params = load_params('params.yaml')

        model_info_path = params['model_evaluation']['experiment_info_path']
        model_name = params['model_evaluation']['model_name']

        setup_mlflow(params['mlflow_config'])

        model_info = load_model_info(model_info_path)

        register_model(model_name,model_info)

        logging.info("Model registerating compoenet ran dsuccessfully")

    except Exception as e:
        print(f'Error : {e}')
        raise e
if __name__ == '__main__':
    main()