import sys
sys.path.append("../src/")
from model_dispatcher import MODEL_DISPATCHER

model = MODEL_DISPATCHER["resnet34"](pretrained=False)
print(model)