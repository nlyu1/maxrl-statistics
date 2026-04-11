import torch
from jaxtyping import Float
from torch import Tensor


def zero_mean_rsq_score(
    *,
    prediction: Float[Tensor, "batch"],
    target: Float[Tensor, "batch"],
) -> Float[Tensor, ""]:
    prediction = prediction.float()
    target = target.float()

    ss_res = torch.sum((target - prediction) ** 2)
    ss_tot = torch.sum(target**2)
    return 1 - ss_res / ss_tot.clamp_min(torch.finfo(target.dtype).eps)
