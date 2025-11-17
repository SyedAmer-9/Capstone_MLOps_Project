import unittest
import mlflow
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from scipy.sparse import load_npz
import yaml


class TestModelLoading(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - runs once before all tests"""
        
        # Load params to get model name
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        
        # Set up DagHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_USER_TOKEN environment variable is not set")
        
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        
        dagshub_url = 'https://dagshub.com'
        repo_owner = 'SyedAmer-9'
        repo_name = 'Capstone_MLOps_Project'
        
        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
        
        # Load the model from MLflow model registry
        cls.model_name = params['model_evaluation']['model_name']  # 'sentiment-classifier'
        cls.model_version = cls.get_latest_model_version(cls.model_name, stage="Staging")
        
        if not cls.model_version:
            raise ValueError(f"No model found in Staging for {cls.model_name}")
        
        cls.model_uri = f'models:/{cls.model_name}/{cls.model_version}'
        print(f"Loading model: {cls.model_uri}")
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)
        
        # Load vectorizer
        vectorizer_path = params['feature_engineering']['models_path'] + '/' + \
                         params['feature_engineering']['vectorizer_file_name']
        
        with open(vectorizer_path, 'rb') as f:
            cls.vectorizer = pickle.load(f)
        
        # Load holdout test data (sparse format)
        processed_path = params['feature_engineering']['processed_data_path']
        cls.X_test = load_npz(os.path.join(processed_path, 'X_test.npz'))
        cls.y_test = np.load(os.path.join(processed_path, 'y_test.npy'))
        
        print(f"Loaded test data: X_test shape = {cls.X_test.shape}, y_test shape = {cls.y_test.shape}")
    
    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        """Get the latest model version from MLflow registry"""
        try:
            client = mlflow.MlflowClient()
            latest_versions = client.get_latest_versions(model_name, stages=[stage])
            return latest_versions[0].version if latest_versions else None
        except Exception as e:
            print(f"Error getting latest model version: {e}")
            return None
    
    def test_01_model_loaded_properly(self):
        """Test that model object is loaded successfully"""
        self.assertIsNotNone(self.model, "Model should be loaded successfully")
        print(f"✓ Model loaded: {self.model_name} v{self.model_version}")
    
    def test_02_vectorizer_loaded(self):
        """Test that vectorizer is loaded successfully"""
        self.assertIsNotNone(self.vectorizer, "Vectorizer should be loaded")
        self.assertTrue(hasattr(self.vectorizer, 'transform'), "Vectorizer should have transform method")
        print(f"✓ Vectorizer loaded with {len(self.vectorizer.get_feature_names_out())} features")
    
    def test_03_model_signature(self):
        """Test model input/output signature"""
        # Create sample input
        input_text = "This is a great movie"
        input_transformed = self.vectorizer.transform([input_text])
        
        # Convert sparse matrix to DataFrame (MLflow pyfunc format)
        input_df = pd.DataFrame(
            input_transformed.toarray(),
            columns=[str(i) for i in range(input_transformed.shape[1])]
        )
        
        # Predict
        prediction = self.model.predict(input_df)
        
        # Verify shapes
        self.assertEqual(
            input_df.shape[1], 
            len(self.vectorizer.get_feature_names_out()),
            "Input features should match vectorizer dimensions"
        )
        self.assertEqual(
            len(prediction), 
            input_df.shape[0],
            "Number of predictions should match number of inputs"
        )
        self.assertIn(
            prediction[0], 
            [0, 1],
            "Prediction should be binary (0 or 1)"
        )
        
        print(f"✓ Model signature validated: Input shape={input_df.shape}, Output={prediction[0]}")
    
    def test_04_model_performance(self):
        """Test model meets minimum performance thresholds on holdout data"""
        
        # Convert sparse matrix to DataFrame for MLflow pyfunc
        X_test_df = pd.DataFrame(
            self.X_test.toarray(),
            columns=[str(i) for i in range(self.X_test.shape[1])]
        )
        
        # Predict using the model
        y_pred = self.model.predict(X_test_df)
        
        # Calculate performance metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        
        print(f"\n📊 Model Performance Metrics:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        
        # Define minimum acceptable thresholds
        MIN_ACCURACY = 0.60
        MIN_PRECISION = 0.55
        MIN_RECALL = 0.55
        MIN_F1 = 0.55
        
        # Assert performance meets thresholds
        self.assertGreaterEqual(
            accuracy, MIN_ACCURACY,
            f'Accuracy {accuracy:.4f} should be at least {MIN_ACCURACY}'
        )
        self.assertGreaterEqual(
            precision, MIN_PRECISION,
            f'Precision {precision:.4f} should be at least {MIN_PRECISION}'
        )
        self.assertGreaterEqual(
            recall, MIN_RECALL,
            f'Recall {recall:.4f} should be at least {MIN_RECALL}'
        )
        self.assertGreaterEqual(
            f1, MIN_F1,
            f'F1 score {f1:.4f} should be at least {MIN_F1}'
        )
        
        print(f"✓ All performance thresholds met!")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)