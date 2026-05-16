import pandas as pd

# טעינת הקובץ לתוך DataFrame
df = pd.read_parquet("data/yellow_tripdata_2024-01.parquet")

# הצגת המבנה והעמודות
print(df.info())
print(df.head())