import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="NLU System API Gateway", version="1.0")

# Define the base URLs for the spaCy and Rasa services
SPACY_BASE_URL = "http://spacy:5001"

# Request models
class DatasetRequest(BaseModel):
    dataset_name: str
    config_name: Optional[str] = None

class TextRequest(BaseModel):
    text: str
    dataset_name: str

# API Endpoints for spaCy-related requests
@app.post("/convert_dataset/spacy")
async def convert_dataset_spacy(request: DatasetRequest):
    try:
        response = requests.post(f"{SPACY_BASE_URL}/convert_dataset", json=request.dict())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/spacy")
async def train_spacy_model(request: DatasetRequest):
    try:
        response = requests.post(f"{SPACY_BASE_URL}/train", json=request.dict())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/spacy")
async def test_spacy_model(request: TextRequest):
    try:
        response = requests.post(f"{SPACY_BASE_URL}/test", json=request.dict())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inspect_dataset/spacy")
async def inspect_dataset(request: DatasetRequest):
    try:
        response = requests.post(f"{SPACY_BASE_URL}/inspect_dataset", json=request.dict())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))