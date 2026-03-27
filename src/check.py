import pandas as pd
from pathlib import Path

data_dir = Path(r"C:\Users\Dragroyale\Desktop\content\vscode\siem-ai-q1\data\raw\CICIDS2017")

csv_files = list(data_dir.glob("*.csv"))
print("Found files:", len(csv_files))

dfs = []
for file in csv_files:
    print("Loading:", file.name)
    df = pd.read_csv(file, low_memory=False)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print("Shape:", data.shape)
print(data.columns.tolist())
print(data.head())