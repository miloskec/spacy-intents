from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dataset_converter import convert_to_spacy
from spacy_trainer import train_spacy
from spacy_tester import test_spacy_model_endpoint
from dataset_logger import inspect_dataset

app = FastAPI(title="spaCy Service", version="1.0")

class DatasetRequest(BaseModel):
    dataset_name: str
    config_name: Optional[str] = None

class TextRequest(BaseModel):
    text: str
    dataset_name: str

@app.post("/convert_dataset")
async def convert_dataset(request: DatasetRequest):
    try:
        convert_to_spacy(request.dataset_name, request.config_name)
        return {"message": f"Dataset {request.dataset_name} converted to spaCy format."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_spacy_model_endpoint(request: DatasetRequest):
    try:
        train_spacy(request.dataset_name)
        return {"message": f"spaCy model trained with dataset {request.dataset_name}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/inspect_dataset")
async def inspect_ds(request: DatasetRequest):
    try:
        inspect_dataset(request.dataset_name, request.config_name)
        return {"message": f"Dataset {request.dataset_name} format logged."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test")
async def test_spacy_endpoint(request: TextRequest):
    try:
        result = test_spacy_model_endpoint(request.text, request.dataset_name)
        return {"intents": result.get("intents"), "entities": result.get("entities")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
