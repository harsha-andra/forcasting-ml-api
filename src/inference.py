import torch
import pandas as pd
import numpy as np
from src.model import ForecastingModel
from src.data_processing import load_and_preprocess_data

def load_model(model_path, embedding_sizes, num_numerical_features):
    model = ForecastingModel(embedding_sizes, num_numerical_features)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_single_row(model, row, cat_cols, num_cols):
    x_cat = torch.tensor([row[cat_cols].values.astype(np.int64)])
    x_num = torch.tensor([row[num_cols].values.astype(np.float32)])
    with torch.no_grad():
        pred = model(x_cat, x_num)
    return pred.item()

def main():
    print("🔍 Loading processed data...")
    df = load_and_preprocess_data()  # Or remove limit if needed

    cat_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    num_cols = ['lag_1', 'lag_7', 'lag_28', 'rolling_mean_7', 'rolling_mean_28']

    # Use a recent row as test input
    test_row = df[df['d'] > 1000].iloc[0]

    # Define same embedding sizes as during training
    embedding_sizes = [(df[col].nunique(), min(50, (df[col].nunique() + 1) // 2)) for col in cat_cols]

    print("📦 Loading model...")
    model = load_model("models/forecasting_model.pth", embedding_sizes, num_numerical_features=len(num_cols))

    print("🧠 Predicting...")
    prediction = predict_single_row(model, test_row, cat_cols, num_cols)

    print(f"📈 Forecasted sales: {prediction:.2f}")

if __name__ == "__main__":
    main()
