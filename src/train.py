#!/home/bmabir/miniconda/envs/bengali-ai-gpu/bin/python
import os
import ast
from model_dispatcher import MODEL_DISPATCHER
from dataset import BengaliDatasetTrain
import torch
import torch.nn as nn
from tqdm import tqdm
import datetime
import numpy as np

DEVICE = "cuda"
TRAINING_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")

IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH"))
EPOCHS = int(os.environ.get("EPOCHS"))

TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE"))

MODEL_MEAN = ast.literal_eval(os.environ.get("MODEL_MEAN")) #ast.literal_eval --> String list to normal list
MODEL_STD = ast.literal_eval(os.environ.get("MODEL_STD"))  

TRAINING_FOLDS = ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
VALIDATION_FOLDS = ast.literal_eval(os.environ.get("VALIDATION_FOLDS"))
BASE_MODEL = os.environ.get("BASE_MODEL")

# to track the training loss as the model trains
train_losses = []
# to track the validation loss as the model trains
# valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
# avg_valid_losses = [] 

#**********experiment with different types of loss functions
def loss_fn(outputs,targets):
    out1, out2, out3 = outputs
    tartget1, tartget2,tartget3 = targets
    loss1 = nn.CrossEntropyLoss()(out1, tartget1)
    loss2 = nn.CrossEntropyLoss()(out2, tartget2)
    loss3 = nn.CrossEntropyLoss()(out3, tartget3)
    return (loss1+loss2+loss3)/3

def train(dataset,data_loader,model,optimizer):
    model.train()

    global train_losses
    for batch_index, dataSet in tqdm(enumerate(data_loader),total=int(len(dataset)/data_loader.batch_size)):
        image = dataSet["image"]
        grapheme_root = dataSet["grapheme_root"]
        vowel_diacritic = dataSet["vowel_diacritic"]
        consonant_diacritic = dataSet["consonant_diacritic"]

        image = image.to(DEVICE,dtype=torch.float)
        grapheme_root = grapheme_root.to(DEVICE,dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE,dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE,dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(image)
        targets = (grapheme_root,vowel_diacritic,consonant_diacritic)
        loss=loss_fn(outputs,targets)

        loss.backward()
        optimizer.step()
        #record training loss
        train_losses.append(loss.item())
    # return train_losses

def evaluate(dataset,data_loader,model):
    model.eval()

    final_loss = 0
    counter = 0
    for batch_index, dataSet in tqdm(enumerate(data_loader),total=int(len(dataset)/data_loader.batch_size)):
        counter += 1
        image = dataSet["image"]
        grapheme_root = dataSet["grapheme_root"]
        vowel_diacritic = dataSet["vowel_diacritic"]
        consonant_diacritic = dataSet["consonant_diacritic"]

        image = image.to(DEVICE,dtype=torch.float)
        grapheme_root = grapheme_root.to(DEVICE,dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE,dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE,dtype=torch.long)
        with torch.no_grad():
            outputs = model(image)
            targets = (grapheme_root,vowel_diacritic,consonant_diacritic)
            loss=loss_fn(outputs,targets)
            final_loss += loss
    return final_loss / counter


def main():
    global avg_train_losses
    print(f"Training with {BASE_MODEL}")
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)
    model.to(DEVICE)


    train_dataset= BengaliDatasetTrain(
        folds=TRAINING_FOLDS,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    #Validation
    valid_dataset= BengaliDatasetTrain(
        folds=VALIDATION_FOLDS,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    #********** experiment with different optimizers
    optimizer =torch.optim.Adam(model.parameters(),lr=1e-4) # Use can use differential learning rate for different layer and experiment with it
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",
                                                            patience=5,factor=0.3,verbose=True)
    if torch.cuda.device_count() > 1 :
        model = nn.DataParallel(model)
    
    os.makedirs("../input/bengali_models/", exist_ok=True)
    logFile= open("../input/bengali_models/validation_log.txt","a")
    logFile.write(f"\n Run Time: {str(datetime.datetime.now())}")
    #****** can also add Early stopping https://github.com/Bjarten/early-stopping-pytorch
    for epoch in range(EPOCHS):
        train(train_dataset,train_loader,model, optimizer)
        val_score= evaluate(valid_dataset,valid_loader,model)

        avg_train_loss = np.average(train_losses)
        avg_train_losses.append(avg_train_loss)

        schedular.step(val_score)
        print(f"{epoch} Epoch Validation Score: {val_score} avg train loss:{avg_train_loss}")
        logFile.write(f" {BASE_MODEL}_fold{VALIDATION_FOLDS[0]}_{epoch} Epoch Validation Score: {val_score} avg train loss:{avg_train_loss} ")
        torch.save(model.state_dict(),f"../input/bengali_models/{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.bin")
    
    logFile.close()
if __name__ == "__main__":
    main()