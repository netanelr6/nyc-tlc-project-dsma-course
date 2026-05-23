import pandas as pd

# טעינת הקובץ לתוך DataFrame
# df = pd.read_parquet("data/raw/yellow_tripdata_2024-01.parquet")
df = pd.read_parquet ("data\processed\yellow_tripdata_2024-01_clean.parquet")

# הצגת המבנה והעמודות
print(df.info())
print(df.head())