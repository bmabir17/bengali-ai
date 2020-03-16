import models

MODEL_DISPATCHER = {
    'resnet34': models.ResNet34,
    'se_resnet50':models.Se_resnet50,
    'se_resnext50':models.Se_resnext50,
    'se_resnet50_margin':models.Se_resnet50_Margin
}