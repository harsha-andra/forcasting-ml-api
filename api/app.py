from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import pandas as pd
from src.model import ForecastingModel
import json
import os
from pathlib import Path

# Safe path logic
BASE_DIR = Path(__file__).resolve().parent.parent
embedding_path = BASE_DIR / "src" / "models" / "embedding_sizes.json"
print(f"🔍 Looking for: {embedding_path}")




if not os.path.exists(embedding_path):
    raise FileNotFoundError(f"❌ File not found: {embedding_path}")

with open(embedding_path) as f:
    embedding_sizes = json.load(f)

embedding_sizes = [tuple(x) for x in embedding_sizes]

app = FastAPI()

class ForecastInput(BaseModel):
    item_id: int
    dept_id: int
    cat_id: int
    store_id: int
    state_id: int
    lag_1: float
    lag_7: float
    lag_28: float
    rolling_mean_7: float
    rolling_mean_28: float

# Load model
model_path = BASE_DIR / "src" / "models" / "forecasting_model.pth"
model = ForecastingModel(embedding_sizes, num_numerical_features=5)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

@app.post("/predict")
def predict(input: ForecastInput):
    x_cat = torch.tensor([[input.item_id, input.dept_id, input.cat_id, input.store_id, input.state_id]])
    x_num = torch.tensor([[input.lag_1, input.lag_7, input.lag_28, input.rolling_mean_7, input.rolling_mean_28]])
    with torch.no_grad():
        pred = model(x_cat, x_num)
    return {"forecasted_sales": round(pred.item(), 2)}
