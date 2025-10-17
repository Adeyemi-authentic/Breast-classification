import torch
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import torch.nn as nn

# Define the model
class BreastClassifier(nn.Module):
    def __init__(self, input_size,hidden_size,out_dim,dropout=0.4):
        super(BreastClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size,out_dim)
        )

    def forward(self, x):
        return self.model(x)


input_size=30
hidden_size=124
dropout=0.4
out_dim=2

model =BreastClassifier(input_size,hidden_size,out_dim,dropout)
model.load_state_dict(torch.load("breast_cancer_model.pth", map_location=torch.device('cpu')))
model.eval()

# FastAPI app
app=FastAPI()

class BreastFeatures(BaseModel):
    features:List[float]

@app.get("/")
def home():
    return {"message": "Welcome to the Breast Cancer Classification App"}

@app.get("/about")
def about():
    return {"message": "Visit https://github.com/Adeyemi-authentic for more of my works - Author"}

@app.post("/predict")
def predict(data: BreastFeatures):
    if len(data.features) !=input_size:
        return {"error": f"Expected {input_size} features, but got {len(data.features)} features"}


    input_tensor = torch.tensor([data.features],dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class=torch.argmax(output,1).item()

    labels = {0: "benign",1: "malignant"}
    return {"prediction":labels[predicted_class]}
