FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY flask_app/requirements.txt requirements.txt
RUN pip install --default-timeout=100 --retries 5 -r requirements.txt
RUN python -m nltk.downloader stopwords wordnet

# Copy App Code
COPY src/ ./src/
COPY flask_app/ ./flask_app/
COPY params.yaml .

# Copy the Artifact (The file we downloaded in CI)
# We create the folder first to be safe
RUN mkdir -p models
COPY models/vectorizer.pkl ./models/vectorizer.pkl

EXPOSE 5001
CMD ["uvicorn", "flask_app.app:app", "--host", "0.0.0.0", "--port", "5001"]