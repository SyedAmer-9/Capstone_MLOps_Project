import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml
from src.logger import logging
import dill
from scipy.sparse import save_npz


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
   
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logging.info('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def apply_bow_and_save_vectorizer(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    
    try:
        logging.info("Applying BOW...")
        vectorizer = CountVectorizer(max_features=max_features)

        X_train = train_data['review'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['review'].values
        y_test = test_data['sentiment'].values

        X_train_sparse = vectorizer.fit_transform(X_train)
        X_test_sparse = vectorizer.transform(X_test)

        logging.info('Bag of Words applied and data transformed.')

        
        models_path = os.path.join("./models")
        os.makedirs(models_path, exist_ok=True)
        vectorizer_path = os.path.join(models_path, "vectorizer.pkl")
        
        with open(vectorizer_path, 'wb') as f:
            dill.dump(vectorizer, f)
        logging.info(f"Vectorizer saved to {vectorizer_path}")

        
        return X_train_sparse, y_train, X_test_sparse, y_test
    except Exception as e:
        logging.error('Error during Bag of Words transformation: %s', e)
        raise

def save_processed_data(X_train, y_train, X_test, y_test, data_path: str) -> None:
   
    try:
        processed_path = os.path.join(data_path, 'processed')
        os.makedirs(processed_path, exist_ok=True)
        
        # Save the sparse matrices using .npz (NumPy Sparse)
        save_npz(os.path.join(processed_path, "X_train.npz"), X_train)
        save_npz(os.path.join(processed_path, "X_test.npz"), X_test)
        
        # Save the labels as simple numpy arrays
        np.save(os.path.join(processed_path, "y_train.npy"), y_train)
        np.save(os.path.join(processed_path, "y_test.npy"), y_test)
        
        logging.info(f'Processed sparse data saved to {processed_path}')
    except Exception as e:
        logging.error(f'Unexpected error occurred while saving processed data: {e}')
        raise Exception(e) from e

def main():
    try:
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']
        
        
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        
        X_train_sparse, y_train, X_test_sparse, y_test = apply_bow_and_save_vectorizer(train_data, test_data, max_features)

   
        save_processed_data(X_train_sparse, y_train, X_test_sparse, y_test, data_path='./data')
        
    except Exception as e:
        logging.error('Failed to complete the feature engineering process')
        raise Exception(e) from e

if __name__ == '__main__':
    main()