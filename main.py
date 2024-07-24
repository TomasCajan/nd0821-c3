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
bucket_name = "tomsprojectbucket"

# Initialize a session using Amazon S3
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

# Function to download a file from S3 and load it
def download_from_s3(s3, bucket_name, key):
    try:
        response = s3.get_object(Bucket=bucket_name, Key=key)
        print(f"Successfully downloaded {key} from S3")
        return response['Body'].read()
    except Exception as e:
        print(f"Error downloading {key} from S3: {e}")
        return None
    
# Function to read the DVC file and extract the S3 key
def get_s3_key_from_dvc(dvc_file_path):
    with open(dvc_file_path, 'r') as f:
        dvc_data = yaml.safe_load(f)
        hash_value = dvc_data['outs'][0]['md5']
        s3_key = f"files/md5/{hash_value[:2]}/{hash_value[2:]}"
        return s3_key
    
# Detect if running on Heroku
is_heroku = 'DYNO' in os.environ

# Get keys from DVC files
base_path = "app/" if is_heroku else ""

model_dvc_file = os.path.join(base_path, "model", "trained_model.pkl.dvc")
binarizer_dvc_file = os.path.join(base_path, "model", "fitted_binarizer.pkl.dvc")
encoder_dvc_file = os.path.join(base_path, "model", "fitted_encoder.pkl.dvc")

model_key = get_s3_key_from_dvc(model_dvc_file)
binarizer_key = get_s3_key_from_dvc(binarizer_dvc_file)
encoder_key = get_s3_key_from_dvc(encoder_dvc_file)

# Get keys from DVC files >>>>>>>
#model_key = "files/md5/cc/26ee31c12be9df028f09a902254282"
#binarizer_key = "files/md5/2a/e474cfc13b051b7b6091665505b5ba"
#encoder_key = "files/md5/9e/031e77fe1ecfada09f89246d53db42"

model_data = download_from_s3(s3, bucket_name, model_key)
binarizer_data = download_from_s3(s3, bucket_name, binarizer_key)
encoder_data = download_from_s3(s3, bucket_name, encoder_key)

model = pickle.loads(model_data)
encoder = pickle.loads(encoder_data)
lb = pickle.loads(binarizer_data)




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