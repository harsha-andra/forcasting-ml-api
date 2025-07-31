import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from src.data_processing import load_and_preprocess_data
from src.model import ForecastingModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Custom PyTorch Dataset
class SalesDataset(Dataset):
    def __init__(self, df, cat_cols, num_cols, target_col):
        self.cat_data = df[cat_cols].values.astype(np.int64)
        self.num_data = df[num_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.tensor(self.cat_data[idx]), torch.tensor(self.num_data[idx]), torch.tensor(self.targets[idx])

def train_model():
    print("📊 Loading data...")
    df = load_and_preprocess_data()
    df=df[df['d']>1800]
    # Select columns
    cat_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    num_cols = ['lag_1', 'lag_7', 'lag_28', 'rolling_mean_7', 'rolling_mean_28']
    target_col = 'sales'

    # Filter recent days for faster training
    df = df[df['d'] > 1000]  # limit to recent time window

    print("🔀 Splitting data...")
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    # Embedding sizes
    embedding_sizes = [(df[col].nunique(), min(50, (df[col].nunique() + 1) // 2)) for col in cat_cols]

    # Datasets and loaders
    train_ds = SalesDataset(train_df, cat_cols, num_cols, target_col)
    val_ds = SalesDataset(val_df, cat_cols, num_cols, target_col)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    print("🧠 Building model...")
    model = ForecastingModel(embedding_sizes, num_numerical_features=len(num_cols))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    print("🚀 Training...")
    for epoch in range(3):
        model.train()
        train_loss = 0
        for x_cat, x_num, y in train_loader:
            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)

            preds = model(x_cat, x_num)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y.size(0)

        train_loss /= len(train_loader.dataset)

        # Eval
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_cat, x_num, y in val_loader:
                x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
                preds = model(x_cat, x_num)
                loss = criterion(preds, y)
                val_loss += loss.item() * y.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"📈 Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/forecasting_model.pth")
    import json
    with open("models/embedding_sizes.json", "w") as f:
        json.dump(embedding_sizes, f)
    print("✅ Model saved to models/forecasting_model.pth")

if __name__ == "__main__":
    train_model()
