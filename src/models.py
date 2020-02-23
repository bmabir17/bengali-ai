import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F


class ResNet34(nn.Module):
    def __init__(self ,pretrained):
        super(ResNet34,self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)

        # To replace the last layer of the model with these
        self.layer0 = nn.Linear(512,168) # 168 grapheme_root
        self.layer1 = nn.Linear(512,11) # 11 vowel_diacritic
        self.layer2 = nn.Linear(512,7) # 7 consonant_diacritic

    def forward(self,x):
        batch_size ,_,_,_ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size, -1)
        layer0 = self.layer0(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(x)
        return layer0, layer1, layer2 # grapheme_root, vowel_diacritic, consonant_diacritic

