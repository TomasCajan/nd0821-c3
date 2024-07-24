"""
Inference API for Census dataset, a Udacity MLOps Nanodegree capstone project.
This script sets up a FastAPI application for predicting income categories based on Census data using a Logistic Regression model. It handles model and preprocessing object loading from an S3 bucket.
It integrates a full CI/CD process using GitHub Actions (CI) and Heroku (CD).

Author: Tomáš Čajan
Date: 24/7/2024

Endpoints:
- GET /: Health check endpoint, returns a welcome message.
- POST /predict: Accepts JSON input for the Census dataset and returns predictions.

"""
import os
import pandas as pd
import numpy as np
import pickle
import boto3
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uvicorn
from io import BytesIO
from training.ml.data import process_data

app = FastAPI(
    title="Inference API for Census Project",
    description="This API allows users to perform inference on the Census dataset using a trained Logistic Regression model. Use the POST /predict endpoint to get predictions.",
    version="1.0.0"
)


aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
bucket_name = "tomsprojectbucket"

s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

def download_from_s3(s3, bucket_name, key):
    """Function to download a file from S3 and load it"""
    try:
        response = s3.get_object(Bucket=bucket_name, Key=key)
        print(f"Successfully downloaded {key} from S3")
        return response['Body'].read()
    except Exception as e:
        print(f"Error downloading {key} from S3: {e}")
        return None
    
def get_s3_key_from_dvc(dvc_file_path):
    """Function to read the DVC file and extract the S3 key"""
    with open(dvc_file_path, 'r') as f:
        dvc_data = yaml.safe_load(f)
        hash_value = dvc_data['outs'][0]['md5']
        s3_key = f"files/md5/{hash_value[:2]}/{hash_value[2:]}"
        return s3_key
    
# Detect if running on Heroku
is_heroku = 'DYNO' in os.environ
base_path = "/app/" if is_heroku else ""

model_dvc_file = os.path.join(base_path, "model", "trained_model.pkl.dvc")
binarizer_dvc_file = os.path.join(base_path, "model", "fitted_binarizer.pkl.dvc")
encoder_dvc_file = os.path.join(base_path, "model", "fitted_encoder.pkl.dvc")

model_key = get_s3_key_from_dvc(model_dvc_file)
binarizer_key = get_s3_key_from_dvc(binarizer_dvc_file)
encoder_key = get_s3_key_from_dvc(encoder_dvc_file)

model_data = download_from_s3(s3, bucket_name, model_key)
binarizer_data = download_from_s3(s3, bucket_name, binarizer_key)
encoder_data = download_from_s3(s3, bucket_name, encoder_key)

model = pickle.loads(model_data)
encoder = pickle.loads(encoder_data)
lb = pickle.loads(binarizer_data)

class InferenceData(BaseModel):
    data: List[Dict[str, Any]] = Field(
        ...,
        example=[
            {
                "age": 25,
                "workclass": "State-gov",
                "fnlwgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "Black",
                "sex": "Female",
                "capital-gain": 2234,
                "capital-loss": 0,
                "hours-per-week": 20,
                "native-country": "United-States"
            }
        ]
    )

class PredictionResult(BaseModel):
    predictions: List[str] = Field(
        ...,
        example=["<=50K"]
    )

@app.get("/", response_model=Dict[str, str])
async def greet() -> Dict[str, str]:
    """
    Greet the user and provide information about the API.
    """
    return {"message": "Welcome to the Inference API. Use the POST /predict endpoint to get predictions."}

@app.post("/predict", response_model=PredictionResult)
async def predict(inference_data: InferenceData) -> PredictionResult:
    """
    Perform inference on the input data.
    """
    try:
        input_df = pd.DataFrame(inference_data.data)
        cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        
        X, _, _, _ = process_data(input_df, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb)
        
        preds = model.predict(X)
        preds = lb.inverse_transform(preds)
        
        return PredictionResult(predictions=preds.tolist())
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
