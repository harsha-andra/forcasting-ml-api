import pandas as pd
import numpy as np

def load_and_preprocess_data(csv_path='data/sales_train_validation.csv', max_rows=None):
    print("📥 Loading dataset...")
    df = pd.read_csv("C:/Users/harsh/OneDrive/Desktop/projects/forecasting/data/sales_train_validation.csv", nrows=max_rows)


    # Drop id column
    df.drop('id', axis=1, inplace=True)

    # Melt from wide to long
    print("🔄 Reshaping...")
    df_long = df.melt(id_vars=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                      var_name='d',
                      value_name='sales')

    # Convert 'd' to integer day index
    df_long['d'] = df_long['d'].str.extract(r'(\d+)').astype(int)

    # Sort by hierarchy + time
    df_long = df_long.sort_values(['item_id', 'store_id', 'd']).reset_index(drop=True)

    # Generate lag features (lag-1, lag-7, lag-28)
    print("➕ Creating lag features...")
    for lag in [1, 7, 28]:
        df_long[f'lag_{lag}'] = df_long.groupby(['item_id', 'store_id'])['sales'].shift(lag)

    # Rolling means
    print("➕ Creating rolling means...")
    for win in [7, 28]:
        df_long[f'rolling_mean_{win}'] = df_long.groupby(['item_id', 'store_id'])['sales']\
                                                  .transform(lambda x: x.shift(1).rolling(win).mean())

    # Fill missing lag values (first few days) with 0
    df_long.fillna(0, inplace=True)

    # Encode categorical variables with index
    print("🔢 Encoding categories...")
    for col in ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']:
        df_long[col] = df_long[col].astype('category').cat.codes

    print("✅ Data preprocessing complete.")
    return df_long
