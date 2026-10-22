import torch.nn as nn
from torchvision.models import (
    vgg16, VGG16_Weights,
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    efficientnet_b3, EfficientNet_B3_Weights
)

NUM_CLASSES = 5


def get_vgg16(num_classes=NUM_CLASSES):
    model = vgg16(weights=VGG16_Weights.DEFAULT)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model


def get_resnet50(num_classes=NUM_CLASSES):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, num_classes)
    return model


def get_densenet121(num_classes=NUM_CLASSES):
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(1024, num_classes)
    return model


def get_efficientnet_b3(num_classes=NUM_CLASSES):
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(1536, num_classes)
    return model


def get_model(model_name, num_classes=NUM_CLASSES):
    if model_name == "vgg16":
        return get_vgg16(num_classes)
    elif model_name == "resnet50":
        return get_resnet50(num_classes)
    elif model_name == "densenet121":
        return get_densenet121(num_classes)
    elif model_name == "efficientnet_b3":
        return get_efficientnet_b3(num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")