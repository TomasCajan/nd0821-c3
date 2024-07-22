import os
import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uvicorn
from training.ml.data import process_data

app = FastAPI(
    title="Inference API for Census Project",
    description="This API allows users to perform inference on the Census dataset using a trained Logistic Regression model. Use the POST /predict endpoint to get predictions.",
    version="1.0.0"
)

model_path = os.path.join(os.path.dirname(__file__), "model", "trained_model.pkl")
encoder_path = os.path.join(os.path.dirname(__file__), "model", "fitted_encoder.pkl")
binarizer_path = os.path.join(os.path.dirname(__file__), "model", "fitted_binarizer.pkl")

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(encoder_path, 'rb') as file:
    encoder = pickle.load(file)

with open(binarizer_path, 'rb') as file:
    lb = pickle.load(file)

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
    uvicorn.run(app, host="0.0.0.0", port=8000)