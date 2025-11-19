# 1. Match your local Python version (3.12) to prevent Pickle errors
FROM python:3.12-slim

WORKDIR /app

# Environment variables
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements
COPY flask_app/requirements.txt requirements.txt
RUN pip install --default-timeout=100 --retries 5 -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet

# Copy necessary files
COPY src/ ./src/
COPY flask_app/ ./flask_app/
COPY params.yaml .
# Make sure this vectorizer path matches where your script saves it!
# If it's in 'models/', use this:
RUN mkdir -p models
COPY models/vectorizer.pkl ./models/vectorizer.pkl

# Expose port
EXPOSE 5001

# Run with uvicorn
# This is the command to start your Sentiment Analysis API
CMD ["uvicorn", "flask_app.app:app", "--host", "0.0.0.0", "--port", "5001"]