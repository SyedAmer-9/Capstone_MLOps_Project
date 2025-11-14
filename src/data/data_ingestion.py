import pandas as pd
import numpy as np

import os
from sklearn.model_selection import train_test_split
import yaml
from src.logger import logging
from src.connections import s3_connection
from src.connections.s3_connection import s3_operations

from dotenv import load_dotenv
load_dotenv()

def load_params(params_path:str)->dict:
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Paramenters retrieved from %s',params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found : %s',params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error : %s',e)
    except Exception as e:
        logging.error("Unexpected error : %s",e)
        raise

def load_data(data_url:str)-> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logging.info('Data loaded from %s',data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error("Failed to parse the CSV file : %s",e)
    except Exception as e:
        logging.error("Unexpected error occurred while loading the data : %s",e)
        raise

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    try:
        logging.info("Pre processing...")
        final_df = df[df['sentiment'].isin(['positive','negative'])].copy()
        sentiment_map = {'positive': 1, 'negative': 0}
        final_df['sentiment'] = final_df['sentiment'].map(sentiment_map)
        logging.info('Data preprocesing completed')
        return final_df
    except KeyError as e:
        logging.error("Missing column in the dataframe : %s",e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing : %s',e)
        raise

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        logging.info('Train and test data saved to %s',raw_data_path)

    except Exception as e:
        logging.error("Unexpected error occurred while saving the data : %s",e)
        raise

def main():
    try:
        # params = load_params(params_path='params.yaml')
        params = load_params(params_path='params.yaml')
        bucket_name = params['data_ingestion']['s3_bucket']
        file_key = params['data_ingestion']['s3_file_key']
        test_size=params['data_ingestion']['test_size']

        s3=s3_operations()
        logging.info("Loading data from S3...")
        df = s3.fetch_file_from_s3(file_key=file_key, bucket_name=bucket_name)
        
        # test_size = 0.2

        # df = load_data(data_url='https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/data.csv')

        final_df = preprocess_data(df)
        train_data,test_data = train_test_split(final_df,test_size=test_size,random_state=42)
        save_data(train_data,test_data,data_path='./data')
    except Exception as e:
        logging.error("Failed to complete the data ingestion process : %s",e)
        print("Error",{e})
if __name__ == '__main__':
    main()