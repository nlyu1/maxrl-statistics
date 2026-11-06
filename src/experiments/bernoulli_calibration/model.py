import math

from torch import Tensor, nn

from src.experiments.bernoulli_calibration.config import DatasetConfig, RunConfig


class BagProbabilityModel(nn.Module):
    def __init__(self, *, data: DatasetConfig, config: RunConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(data.vocabulary), config.embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.normalization = math.sqrt(data.length)
        nn.init.normal_(self.embedding.weight, std=0.02)
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.bias)

    def forward(self, *, tokens: Tensor) -> Tensor:
        features = self.embedding(tokens).sum(dim=1) / self.normalization
        return self.head(features).squeeze(-1)
