from torch import nn
from models import resnet
# from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .module import MCDropout, GaussianDropout, GaussianMCDropout, \
    MeanAbsoluteError, LambdaLayer, KLDivDistillationLoss, \
    L1DistillationLoss, SmoothL1DistillationLoss, \
    ReverseCrossEntropyLoss, SymmetricCrossEntropyLoss, \
    GeneralizedCrossEntropyLoss, CrossEntropyDistillationLoss, \
    Normalize2D
import torchvision


def get_model(architecture, num_classes, pretrained=False) -> nn.Module:
    match architecture:
        case "resnet18":
            model = resnet.resnet18(pretrained=False, in_channels=3, num_classes=num_classes)
        case "resnet34":
            model = resnet.resnet34(pretrained=False, in_channels=3, num_classes=num_classes)
        case "resnet50":
            model = resnet.resnet50(pretrained=False, in_channels=3, num_classes=num_classes)
        case "resnet101":
            model = resnet.resnet101(pretrained=False, in_channels=3, num_classes=num_classes)
        case "resnet152":
            model = resnet.resnet152(pretrained=False, in_channels=3, num_classes=num_classes)
        case "resnet18_torchvision":
            model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            model.fc = nn.Linear(512, num_classes)
        case "resnet34_torchvision":
            model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            model.fc = nn.Linear(512, num_classes)
        case "resnet50_torchvision":
            model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            model.fc = nn.Linear(2048, num_classes)
        case "resnet101_torchvision":
            model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
            model.fc = nn.Linear(2048, num_classes)
        case "resnet152_torchvision":
            model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None)
            model.fc = nn.Linear(2048, num_classes)
        case "resnet34_dropout":
            model = resnet.resnet34(pretrained=False, in_channels=3, num_classes=num_classes, dropout=nn.Dropout(0.5))
        case _:
            raise NotImplementedError
    return model