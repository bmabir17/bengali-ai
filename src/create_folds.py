import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
# import numpy as np

if __name__ == "__main__":
    df = pd.read_csv("../input/train.csv")
    print(df.head())
    df.loc[:,'kfold'] = -1

    #shuffle dataset

    df = df.sample(frac=1).reset_index(drop=True)


    #for image_id column
    X = df.image_id.values
    # as this is multi-label classification
    y = df[["grapheme_root","vowel_diacritic","consonant_diacritic"]].values #as numpy arrray

    mskf = MultilabelStratifiedKFold(n_splits=5) #into 5 k-fold data

    for fold, (trn_,val_) in enumerate(mskf.split(X,y)):
        print("Train: ",trn_, "VAL:",val_)
        df.loc[val_,"kfold"] = fold
    
    print(df.kfold.value_counts())
    df.to_csv("../input/train_folds.csv",index=False)