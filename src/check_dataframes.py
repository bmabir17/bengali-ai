import pandas as pd 

df = pd.read_parquet("../input/train_image_data_0.parquet")
print(df.head())
