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

#model_path = os.path.join(os.path.dirname(__file__), "model", "trained_model.pkl")
#encoder_path = os.path.join(os.path.dirname(__file__), "model", "fitted_encoder.pkl")
#binarizer_path = os.path.join(os.path.dirname(__file__), "model", "fitted_binarizer.pkl")

#with open(model_path, 'rb') as file:
#    model = pickle.load(file)

#with open(encoder_path, 'rb') as file:
#    encoder = pickle.load(file)

#with open(binarizer_path, 'rb') as file:
#    lb = pickle.load(file)

#try2

#def load_file(paths):
#    for path in paths:
#        try:
#            with open(path, 'rb') as file:
#                print(f"Successfully loaded file from: {path}")
#                return pickle.load(file)
#        except FileNotFoundError:
#            print(f"File not found at: {path}")
#    raise FileNotFoundError(f"File not found in any of the provided paths: {paths}")

# Define potential paths for the model, encoder, and binarizer
#model_paths = [
#    os.path.join(os.path.dirname(__file__), "model", "trained_model.pkl"),
#    os.path.join("/app", "model", "trained_model.pkl")
#]

#encoder_paths = [
#    os.path.join(os.path.dirname(__file__), "model", "fitted_encoder.pkl"),
#    os.path.join("/app", "model", "fitted_encoder.pkl")
#]
#
#binarizer_paths = [
#    os.path.join(os.path.dirname(__file__), "model", "fitted_binarizer.pkl"),
#    os.path.join("/app", "model", "fitted_binarizer.pkl")
#]#

# Load model, encoder, and binarizer
#model = load_file(model_paths)
#encoder = load_file(encoder_paths)
#lb = load_file(binarizer_paths)

# Define AWS credentials and S3 bucket details
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
s3_bucket_name = 'tomsprojectbucket'

# Initialize a session using Amazon S3
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

# Function to download a file from S3 and load it
def download_from_s3(bucket_name, key, download_path):
    try:
        s3.download_file(bucket_name, key, download_path)
        print(f"Successfully downloaded {key} from S3 to {download_path}")
    except Exception as e:
        raise RuntimeError(f"Error downloading {key} from S3: {e}")

# Function to get S3 key from DVC file
def get_s3_key_from_dvc_file(dvc_file_path):
    with open(dvc_file_path, 'r') as file:
        dvc_data = yaml.safe_load(file)
        md5_hash = dvc_data['outs'][0]['md5']
        s3_key = f'files/md5/{md5_hash[:2]}/{md5_hash[2:]}'
        return s3_key

# Get keys from DVC files
model_key = get_s3_key_from_dvc_file('model/trained_model.pkl.dvc')
encoder_key = get_s3_key_from_dvc_file('model/fitted_encoder.pkl.dvc')
binarizer_key = get_s3_key_from_dvc_file('model/fitted_binarizer.pkl.dvc')

# Download and load the files
def load_file_from_s3(bucket_name, key):
    download_path = f'/tmp/{os.path.basename(key)}'
    download_from_s3(bucket_name, key, download_path)
    with open(download_path, 'rb') as file:
        return pickle.load(file)

model = load_file_from_s3(s3_bucket_name, model_key)
encoder = load_file_from_s3(s3_bucket_name, encoder_key)
lb = load_file_from_s3(s3_bucket_name, binarizer_key)




class InferenceData(BaseModel):
    data: List[Dict[str, Any]]

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

@app.get("/", response_model=Dict[str, str])
async def greet() -> Dict[str, str]:
    """
    Greet the user and provide information about the API.
    """
    return {"message": "Welcome to the Inference API. Use the POST /predict endpoint to get predictions."}

@app.post("/predict", response_model=Dict[str, List[str]])
async def predict(inference_data: InferenceData) -> Dict[str, List[str]]:
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
        
        return {"predictions": preds.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)