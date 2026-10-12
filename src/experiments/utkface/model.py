import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18

from src.experiments.utkface.config import Head
from src.experiments.utkface.losses import MAX_AGE, NUM_BINS


class ScalarHead(nn.Module):
    def __init__(self, *, features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(features, 1)

    def forward(self, features: Tensor) -> Tensor:
        values = self.linear(features)
        values = values.double() if values.dtype == torch.float64 else values.float()
        return values.sigmoid() * MAX_AGE


class DigitHead(nn.Module):
    def __init__(self, *, features: int) -> None:
        super().__init__()
        self.tens = nn.Linear(features, 10)
        self.condition = nn.Linear(features, 128)
        self.embedding = nn.Embedding(10, 128)
        self.units = nn.Linear(128, 10)

    def forward(self, features: Tensor) -> Tensor:
        tens = self.tens(features).float().log_softmax(-1)
        hidden = self.condition(features)[:, None, :] + self.embedding.weight[None]
        units = self.units(hidden.relu()).float().log_softmax(-1)
        return (tens[:, :, None] + units).flatten(1)


def build_model(*, pretrained: bool, head: Head) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    features = model.fc.in_features
    if head == "categorical":
        model.fc = nn.Linear(features, NUM_BINS)
    elif head == "scalar":
        model.fc = ScalarHead(features=features)
    elif head == "digits":
        model.fc = DigitHead(features=features)
    else:
        raise ValueError(f"Unsupported head: {head}")
    return model
