import numpy as np
import pandas as pd
import os
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
from src.logger import logging
from scipy.sparse import load_npz

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

def load_processed_data(processed_data_path: str) -> tuple:
    try:
        logging.info(f"Loading processed TRAIN data from {processed_data_path}")
        X_train = load_npz(os.path.join(processed_data_path, "X_train.npz"))
        y_train = np.load(os.path.join(processed_data_path, "y_train.npy"))
        logging.info("Processed TRAIN data loaded successfully.")
        return X_train, y_train
    except Exception as e:
        logging.error(f'Error loading processed data: {e}')
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray,hyperparameters:dict) -> LogisticRegression:
    try:
        clf = LogisticRegression(**hyperparameters)
        clf.fit(X_train, y_train)
        logging.info('Model training completed')
        return clf
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        params = load_params('params.yaml')

        hyperparams = params['model_training']['LogisticRegression']['params']

        processed_data_path = params['feature_engineering']['processed_data_path']
        model_save_path = params['model_training']['model_save_path']

        model_file_path = os.path.join(model_save_path, "model.pkl")

        logging.info(f"Loading training data...")
        X_train, y_train = load_processed_data(processed_data_path)

        clf = train_model(X_train, y_train, hyperparams)        
        save_model(clf, model_file_path)
        logging.info("Model training stage complete.")
    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()