FROM python:3.10-slim

WORKDIR /app

ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_NO_CACHE_DIR=1

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --default-timeout=100 --retries 5 -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

COPY flask_app/ /app/

COPY models/vectorizer.pkl /app/models/vectorizer.pkl




EXPOSE 5001

#local
# CMD ["python", "app.py"]  

#Prod
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--timeout", "120", "app:app"]