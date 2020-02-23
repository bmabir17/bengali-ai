import pandas as pd


class BengaliDatasetTrain:
    def __init__(self,folds, img_height,img_width, mean, std):
        df = pd.read_csv("../input/train_folds.csv")
        df = df[["image_id","grapheme_root","vowel_diacritic","consonant_diacritic","kfold"]]

        # Getting label data from train_folds.csv
        df = df[df.kfold.isin(folds)].reset_index(drop=True)
        self.image_ids = df.image_id.values
        self.grapheme_root = df.grapheme_root.values
        self.vowel_diacritic = df.vowel_diacritic.values
        self.consonant_diacritic = df.consonant_diacritic.values
        

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self,item):
        image = joblib.load(f"../input/image_pickles/{self.image_ids[item]}.pkl")