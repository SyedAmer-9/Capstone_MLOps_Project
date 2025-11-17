import os
import mlflow

def promote_model():
    dagshub_token = os.getenv('DAGSHUB_USER_TOKEN')
    
    if not dagshub_token:
        raise EnvironmentError("Environment variable is not set")
    
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = 'https://dagshub.com'
    repo_owner = 'SyedAmer-9'
    repo_name='Capstone_MLOps_Project'

    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

    client = mlflow.MlflowClient()

    model_name = 'sentiment-classifier'

    latest_version_staging = client.get_latest_versions(model_name,stages=["Staging"])[0].version

    prod_versions = client.get_latest_versions(model_name,stages=['Production'])

    for version in prod_versions:
        client.transition_model_version_stage(
            name = model_name,
            version=version.version,
            stage="Archived"
        )
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage='Production'
    )

    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == '__main__':
    promote_model()