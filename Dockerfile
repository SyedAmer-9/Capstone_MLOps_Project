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

# --- DVC INTEGRATION FIX ---
# Install AWS CLI inside the image for DVC to talk to S3
RUN pip install awscli

# COPY the DVC configuration files
COPY .dvc/ .dvc/
COPY .dvcignore .dvcignore
COPY dvc.yaml dvc.yaml
COPY dvc.lock dvc.lock

# 1. PULL the Vectorizer (models/vectorizer.pkl) from S3 during the build
# This requires that your GitHub Actions configured AWS credentials
# before running docker build.
RUN dvc pull models/vectorizer.pkl
# --- END DVC INTEGRATION FIX ---

# Copy app files
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