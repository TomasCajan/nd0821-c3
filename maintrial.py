import os
import pandas as pd
import numpy as np
import pickle
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uvicorn
from training.ml.data import process_data

app2 = FastAPI(
    title="Inference API for Census Project",
    description="This API allows users to perform inference on the Census dataset using a trained Logistic Regression model. Use the POST /predict endpoint to get predictions.",
    version="1.0.0"
)



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

@app2.get("/", response_model=Dict[str, str])
async def greet() -> Dict[str, str]:
    """
    Greet the user and provide information about the API.
    """
    return {"message": "Welcome to the Inference API. Use the POST /predict endpoint to get predictions."}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app2, host="0.0.0.0", port=port)