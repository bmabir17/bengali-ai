import pretrainedmodels
import glob
import torch
import albumentations
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
from tqdm import tqdm
from PIL import Image
import joblib
import torch.nn as nn
from torch.nn import functional as F

from model_dispatcher import MODEL_DISPATCHER
from dataset import BengaliDatasetTest

TEST_BATCH_SIZE = 32
MODEL_MEAN=(0.485,0.465,0.406)
MODEL_STD=(0.229,0.224,0.225)
IMG_HEIGHT=137
IMG_WIDTH=236
DEVICE="cuda"
BASE_MODEL="resnet34"
# def calc_auc(pred):
#     _,
# def calc_accuracy(pred,label):


def predict(model):
    model.to(DEVICE)
    model.eval()
    predictions=[]
    files = glob.glob("../input/bengaliai-cv19/test_image_data_*.parquet")
    for file_idx, f in enumerate(files):
        df = pd.read_parquet(f)
        dataset = BengaliDatasetTest(df=df,img_height=IMG_HEIGHT,img_width=IMG_WIDTH,
                                    mean=MODEL_MEAN,std=MODEL_STD)
        data_loader = torch.utils.data.DataLoader(dataset,batch_size=TEST_BATCH_SIZE,shuffle=False,num_workers=4)
        for bi , d in enumerate(data_loader):
            image = d["image"]
            img_id= d["image_id"]
            image= image.to(DEVICE,dtype=torch.float)
            g,v,c=model(image)
            g = np.argmax(g,axis=1) 
            v = np.argmax(v,axis=1)
            c = np.argmax(c,axis=1)
            for i,im_id in enumerate(img_id):
                predictions.append((f"{im_id}_grapheme_root",g[i]))
                predictions.append((f"{im_id}_vowel_diacritic",v[i]))
                predictions.append((f"{im_id}_consonant_diacritic",c[i]))
    return predictions

def main():
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)
    model.load_state_dict(torch.load("../input/bengali_models/resnet34_fold4.bin")) # load dataparallel if trained the model using multiple gpus

    


    predictions=predict(model)
    print(predictions)
    # df = pd.read_csv("../input/test.csv")
    # #for image_id column
    # img_id = df.image_id.values
    # component = df.component.values


    sub = pd.DataFrame(predictions,columns=["row_id","target"])
    sub.to_csv("submission.csv",index=False)


main()