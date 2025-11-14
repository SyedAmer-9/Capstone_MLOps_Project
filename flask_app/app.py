import sys
import os
import re
import string
import time
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import dagshub
import yaml
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv

from fastapi import FastAPI,Request,Form,BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response,HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from prometheus_client import CollectorRegistry,Histogram,Counter,generate_latest,CONTENT_TYPE_LATEST

from src.logger import logging

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_ROOT)
ENV_FILE_PATH = os.path.join(PROJECT_ROOT, ".env")
PARAMS_FILE_PATH = os.path.join(PROJECT_ROOT, "params.yaml")

load_dotenv()

logging.info('Loaded .env file')

logging.info("Downloading NLTK resources...")
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))
logging.info("NLTK resources loaded globally.")

def load_params(params_path: str = 'params.yaml') -> dict:
    try:
        with open(params_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error loading {params_path}: {e}")
        raise e

PARAMS = load_params()
logging.info("Loaded params.yaml")

def setup_mlflow():
    logging.info("Setting up MLflow connection...")
    dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_USER_TOKEN environment variable is not set. Check your .env file.")
    
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow_config = PARAMS['mlflow_config']
    mlflow.set_tracking_uri(mlflow_config['mlflow_tracking_uri'])
    mlflow.set_experiment(mlflow_config['experiment_name'])
    logging.info(f"MLflow connection established. Experiment: {mlflow_config['experiment_name']}")

setup_mlflow()

logging.info("Initializing Prometheus metrics...")
registry = CollectorRegistry()
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Request latency in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

class PredictionPipeline:
    def __init__(self):
        self.model = None
        self.vectorizer = None

        self.translator = {1: 'positive', 0: 'negative'}
    
        self.vectorizer_path = os.path.join(
            PROJECT_ROOT, # Start from the project's root
            PARAMS['feature_engineering']['models_path'], 
            PARAMS['feature_engineering']['vectorizer_file_name']
        )
        self.model_name = PARAMS['model_evaluation']['model_name']

    def load_model_and_vectorizer(self):
        try:
            logging.info("Loading the model from MLflow Model Registry")
            model_name = PARAMS['model_evaluation']['model_name']

            client = mlflow.MlflowClient()

            latest_versions= client.get_latest_versions(model_name,stages=['Staging'])

            if not latest_versions:
                logging.warning(f"No model in 'Staging'. Falling back to 'Production'.")
                latest_versions = client.get_latest_versions(model_name, stages=["Production"])

            if not latest_versions:
                logging.error(f"CRITICAL: No 'Staging' or 'Production' model found for {model_name}.")
                raise Exception(f"No production-ready model found for {model_name}")
            
            model_uri = f"models:/{model_name}/{latest_versions[0].version}"
            logging.info(f"Loading model from URI: {model_uri}")

            self.model = mlflow.pyfunc.load_model(model_uri)
            logging.info("MLflow model loaded successfully.")

            vectorizer_path = os.path.join(PARAMS['feature_engineering']['models_path'], PARAMS['feature_engineering']['vectorizer_file_name'])

            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            logging.info(f"Vectorizer loaded from {vectorizer_path}")

        except Exception as e:
            logging.error(f"CRITICAL: Failed to load model or vectorizer: {e}")
            raise e
        
    def predict(self, text: str) -> str:
        if not self.model or not self.vectorizer:
            self.load_model_and_vectorizer()
        
        try:
            clean_text=normalize_text(text)

            features_sparse = self.vectorizer.transform([clean_text])

            
            prediction_numeric = self.model.predict(features_sparse)

            result_str = self.translator.get(prediction_numeric[0], "unknown")

            PREDICTION_COUNT.labels(prediction=result_str).inc()

            return result_str
        
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise e
        
def normalize_text(text: str) -> str:
    """Master function to clean a single string."""
    text = removing_urls(text)
    text = removing_numbers(text)
    text = text.lower()
    text = removing_punctuations(text)
    text = remove_stop_words(text) # Uses global STOP_WORDS
    text = lemmatization(text) # Uses global LEMMATIZER
    return text.strip()    

def lemmatization(text):
    return " ".join([LEMMATIZER.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    return " ".join([word for word in text.split() if word not in STOP_WORDS])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def removing_punctuations(text):
   
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "") # <-- THE MISSING LINE
    text = re.sub(r'\s+', ' ', text).strip() # Use r'' for raw string
    return text

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)
app = FastAPI()



templates = Jinja2Templates(directory=os.path.join(APP_ROOT, "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

pipeline = PredictionPipeline()

@app.get('/',response_class=HTMLResponse)

async def home(request:Request):
    REQUEST_COUNT.labels(method="GET",endpoint='/').inc()

    start_time = time.time()

    response = templates.TemplateResponse(
        'index.html',{'request':request,'result':None} 
    )

    REQUEST_LATENCY.labels(endpoint='/').observe(time.time()-start_time)
    return response

@app.post('/predict',response_class=HTMLResponse)

async def predict(request:Request):
    REQUEST_COUNT.labels(method='POST',endpoint='/predict').inc()
    start_time = time.time()

    try:
        form = await request.form()
        text_input = form.get('text')

        if not text_input:
            return templates.TemplateResponse(
                'index.html',{'request':request,'result':"Error:No text provided"}
            )
        prediction = pipeline.predict(text_input)

        return templates.TemplateResponse(
            'index.html',{'request':request,'result':f"Prediction : {prediction.upper()}"}
        )
    except Exception as e:
        logging.exception("Error during /predict")
        return templates.TemplateResponse(
            "index.html", {"request": request, "result": f"Error: {e}"},
        )
    finally:
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

@app.get("/metrics")
def metrics():
    # Expose only our custom Prometheus metrics.
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)  

if __name__ == "__main__":
    try:
        app_host = PARAMS.get('app', {}).get('host', '0.0.0.0')
        app_port = int(PARAMS.get('app', {}).get('port', 5000))
        
        app_run(app, host=app_host, port=app_port)
    except Exception as e:
        logging.error(f"Failed to start app: {e}")
        sys.exit(1)