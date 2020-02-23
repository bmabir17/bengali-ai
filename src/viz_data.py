import sys
sys.path.append("../src/")
import torch
import matplotlib.pyplot as plt 
from dataset import BengaliDatasetTrain
import numpy as np 

dataset = BengaliDatasetTrain(folds=[0,1],img_height=137,img_width=236,
                                mean=(0.485,0.465,0.406),
                                std=(0.229,0.224,0.225))
print(len(dataset))

idx=1
img =dataset[idx]["image"]
print(dataset[idx]["grapheme_root"] )
print(dataset[idx]["vowel_diacritic"] )
print(dataset[idx]["consonant_diacritic"] )
npimg = img.numpy()
plt.imshow(np.transpose(npimg,(1,2,0)))
plt.show()