import torch
from jaxtyping import Float
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch import Tensor


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class CorrelationCounter:
    xy: Float[Tensor, "dim"]
    xx: Float[Tensor, "dim"]
    yy: Float[Tensor, "dim"]

    @dataclass(config=ConfigDict(arbitrary_types_allowed=True))
    class Statistics:
        corr: Float[Tensor, "dim"]
        # Calibrated rsq is just corr**2
        uncalibrated_rsq: Float[Tensor, "dim"]
        beta: Float[Tensor, "dim"]

    @classmethod
    def initialize(
        cls,
        dim: int,
        *,
        device: torch.device | None = None,
    ) -> "CorrelationCounter":
        return cls(
            xy=torch.zeros(dim, device=device),
            xx=torch.zeros(dim, device=device),
            yy=torch.zeros(dim, device=device),
        )

    @torch.no_grad()
    def tick(self, *, x: Float[Tensor, "... dim"], y: Float[Tensor, "... dim"]) -> None:
        """Updates the sufficient statistics"""
        x = x.float().reshape(-1, self.xy.shape[0])
        y = y.float().reshape(-1, self.xy.shape[0])
        self.xy += (x * y).sum(dim=0)
        self.xx += (x * x).sum(dim=0)
        self.yy += (y * y).sum(dim=0)

    def get_stats(self) -> "CorrelationCounter.Statistics":
        """Returns the statistics"""
        eps = torch.finfo(self.xy.dtype).eps
        corr = self.xy / (self.xx * self.yy).sqrt().clamp_min(eps)
        uncalibrated_rsq = (2 * self.xy - self.xx) / self.yy.clamp_min(eps)
        beta = self.xy / self.xx.clamp_min(eps)
        return CorrelationCounter.Statistics(corr=corr, uncalibrated_rsq=uncalibrated_rsq, beta=beta)

    def empty_(self) -> None:
        self.xy.zero_()
        self.xx.zero_()
        self.yy.zero_()
