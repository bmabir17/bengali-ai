import pandas as pd 
import joblib
import glob #to list files
from tqdm import tqdm
import os
### TO create pickles from pandas dataframe (this is to read the data faster while training)

if __name__ == "__main__":
    os.makedirs("path/to/directory", exist_ok=True)
    files = glob.glob("../input/train_*.parquet")
    for f in files:
        df = pd.read_parquet(f) # reading parquet fils as pandas dataframe
        image_ids = df.image_id.values
        df = df.drop("image_id",axis=1)
        image_array = df.values
        for j, img_id in tqdm(enumerate(image_ids),total=len(image_ids)):
            joblib.dump(image_array[j, :], f"../input/image_pickles/{img_id}.pkl")
