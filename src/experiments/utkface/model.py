from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.experiments.utkface.losses import NUM_BINS


def build_model(*, pretrained: bool) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, NUM_BINS)
    return model
