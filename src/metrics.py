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
    n: Float[Tensor, ""]

    @dataclass(config=ConfigDict(arbitrary_types_allowed=True))
    class Statistics:
        corr: Float[Tensor, "dim"]
        mse: Float[Tensor, "dim"]
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
            n=torch.zeros((), device=device),
        )

    @torch.no_grad()
    def tick(self, *, x: Float[Tensor, "... dim"], y: Float[Tensor, "... dim"]) -> None:
        """Updates the sufficient statistics"""
        x = x.float().reshape(-1, self.xy.shape[0])
        y = y.float().reshape(-1, self.xy.shape[0])
        self.xy += (x * y).sum(dim=0)
        self.xx += (x * x).sum(dim=0)
        self.yy += (y * y).sum(dim=0)
        self.n += x.shape[0]

    def get_stats(self) -> "CorrelationCounter.Statistics":
        """Returns the statistics"""
        eps = torch.finfo(self.xy.dtype).eps
        corr = self.xy / (self.xx * self.yy).sqrt().clamp_min(eps)
        mse = (self.xx - 2 * self.xy + self.yy) / self.n.clamp_min(eps)
        beta = self.xy / self.xx.clamp_min(eps)
        return CorrelationCounter.Statistics(corr=corr, mse=mse, beta=beta)

    def empty_(self) -> None:
        self.xy.zero_()
        self.xx.zero_()
        self.yy.zero_()
        self.n.zero_()
