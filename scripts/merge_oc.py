import pandas as pd
import os
from functools import reduce

processed_folder = "data/processed"
output_file = os.path.join(processed_folder, "all_open_close.csv")

files = [f for f in os.listdir(processed_folder) if f.endswith("_open_close.csv")]

dfs = []
for f in files:
    df = pd.read_csv(os.path.join(processed_folder, f))
    dfs.append(df)

# Merge all on 'date'
all_df = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), dfs)
all_df = all_df.sort_values("date").reset_index(drop=True)
all_df.to_csv(output_file, index=False)
print("Saved all_open_close.csv")