from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from joblib import load
import numpy as np

input_size=30
output_class=2

# Load the trained model
rf =load("random_forest_breast_cancer.pkl")

app=FastAPI()
class InputFeatures(BaseModel):
    Features: List[float]


@app.get("/")
def home():
    return {"message": "Welcome to the Breast Cancer Classifier using Random Forest"}


@app.post("/predict")
def predict(data: InputFeatures):
    if len(data.Features) != input_size:
        return {"error": f"We need {input_size} features, but got {len(data.Features)}."}

    # Convert input to 2D array
    input_data=np.array(data.Features).reshape(1, -1)

    # Make prediction
    predicted_class= rf.predict(input_data)[0]

    label= {0: "benign", 1: "malignant"}
    return {"prediction": label[predicted_class]}
