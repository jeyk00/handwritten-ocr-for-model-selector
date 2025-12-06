from contextlib import asynccontextmanager
from typing import List
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from app.model_loader import load_model

# Global variable to hold the model
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model
    ml_models["model"] = load_model()
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

class DigitPayload(BaseModel):
    image: List[List[int]]

@app.post("/predict")
async def predict(payload: DigitPayload):
    model = ml_models["model"]
    
    # Convert input list to numpy array
    input_data = np.array(payload.image, dtype=np.float32)
    
    # Normalize the data (0-255 -> 0.0-1.0)
    input_data = input_data / 255.0
    
    # Convert to tensor and reshape to (B, C, H, W) -> (1, 1, 28, 28)
    # The input is expected to be 28x28
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        
        # Calculate probabilities (model returns log_softmax)
        probabilities = torch.exp(output)
        
        # Get prediction and confidence
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities).item()
        
    return {
        "prediction": prediction,
        "confidence": confidence
    }
