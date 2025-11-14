import numpy as np
import pandas as pd
import os
import re
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logging

nltk.download('wordnet')
nltk.download('stopwords')

LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))
logging.info("NLTK resources loaded globally.")



def preprocess_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove numbers
    text = ''.join([char for char in text if not char.isdigit()])
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuations
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stop words
    text = " ".join([word for word in text.split() if word not in STOP_WORDS])
    
    # Lemmatization
    text = " ".join([LEMMATIZER.lemmatize(word) for word in text.split()])
    return text

def preprocess_dataframe(df, col='review'):
   
    try:
        logging.info(f"Applying preprocessing to '{col}' column...")
        df[col] = df[col].astype(str).apply(preprocess_text)
        df = df.dropna(subset=[col])
        logging.info("Text preprocessing completed.")
        return df
    except KeyError:
        logging.error(f"The required column '{col}' was not found in the DataFrame.")
        raise Exception(f"Missing required column: {col}", sys)
    except Exception as e:
        raise e

def main():
    try:
        train_in_path = './data/raw/train.csv'
        test_in_path = './data/raw/test.csv'
        
        
        processed_data_path = os.path.join('./data', 'interim')
        os.makedirs(processed_data_path, exist_ok=True)


        train_data = pd.read_csv(train_in_path)
        test_data = pd.read_csv(test_in_path)

      
        train_processed_data = preprocess_dataframe(train_data, 'review')
        test_processed_data = preprocess_dataframe(test_data, 'review')

        train_processed_data.to_csv(os.path.join(processed_data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(processed_data_path, "test_processed.csv"), index=False)

        logging.info(f"Processed arrays saved to: {processed_data_path}")

    except Exception as e:
        logging.error('Failed to complete the data transformation process')
        raise Exception(e) from e
if __name__ == '__main__':
    main()