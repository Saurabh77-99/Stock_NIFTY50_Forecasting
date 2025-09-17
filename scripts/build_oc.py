import pandas as pd
import os

raw_folder = "data/raw"
processed_folder = "data/processed"

if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)

def build_open_close(file_path, stock_name):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    daily = df.groupby(df['date'].dt.date).agg(
        { 'open': 'first', 'close': 'last' }
    ).reset_index()
    
    daily.columns = ['date', f'{stock_name}_OPEN', f'{stock_name}_CLOSE']
    daily.to_csv(os.path.join(processed_folder, f"{stock_name}_open_close.csv"), index=False)
    print(f"Saved {stock_name}_open_close.csv")

for file in os.listdir(raw_folder):
    if file.endswith(".csv"):
        stock_name = file.split("_")[0]
        build_open_close(os.path.join(raw_folder, file), stock_name)
