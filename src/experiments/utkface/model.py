import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def create_policy(*, pretrained: bool, seed: int) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    torch.manual_seed(seed)
    model.fc = nn.Linear(512, 100)
    return model
