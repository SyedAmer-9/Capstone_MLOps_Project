import boto3
import pandas as pd
from src.logger import logging
from io import StringIO

class s3_operations:

    def __init__(self,region_name='us-east-1'):

        self.s3_client = boto3.client(
            's3',
            region_name=region_name
        )
        logging.info("Data Ingestion from S3 bucket initialized")

    def fetch_file_from_s3(self,file_key,bucket_name):
        
        try:
            logging.info(f"Fetching file '{file_key}' from S3 bucket '{bucket_name}'...")
            obj = self.s3_client.get_object(Bucket = bucket_name,Key=file_key)
           
           
            df = pd.read_csv(obj['Body'])
            logging.info(f"Succesfully fetched and loaded '{file_key}' from S3 that has {len(df)} records.")
            return df
        except Exception as e:
            logging.exception(f"Failed to fetch '{file_key}' from S3 : {e}")
            return None
        
