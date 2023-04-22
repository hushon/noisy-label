from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152


def get_model(architecture, num_classes):
    if architecture == "resnet18":
        return resnet18(pretrained=False, in_channels=3, num_classes=num_classes)
    elif architecture == "resnet34":
        return resnet34(pretrained=False, in_channels=3, num_classes=num_classes)
    elif architecture == "resnet50":
        return resnet50(pretrained=False, in_channels=3, num_classes=num_classes)
    elif architecture == "resnet101":
        return resnet101(pretrained=False, in_channels=3, num_classes=num_classes)
    elif architecture == "resnet152":
        return resnet152(pretrained=False, in_channels=3, num_classes=num_classes)
    else:
        raise NotImplementedError