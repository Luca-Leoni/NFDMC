import torch
import torch.nn as nn
import torch.nn.functional as F

from ..Archetypes import Flow
from torch import Tensor


class BatchNorm(Flow):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = False):
        super().__init__()

        self.eps = eps
        self.mom = momentum

        self.unconstrained_weight = nn.Parameter(torch.ones(num_features, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(num_features, dtype=torch.float64))

        self.register_buffer("running_mean", torch.zeros_like(self.bias))
        self.register_buffer("running_var", torch.zeros_like(self.bias))


    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        if self.training:
            mean, var = z.mean(0), z.var(0)
            self.running_mean = (1 - self.mom) * self.running_mean + (mean * self.mom)
            self.running_var = (1 - self.mom) * self.running_var + (var * self.mom)
        else:
            mean, var = self.running_mean, self.running_var

        weight = F.softplus(self.unconstrained_weight) + self.eps

        res = weight * (z - mean) / torch.sqrt(var + self.eps) + self.bias
        log_det = weight.log() - 0.5 * torch.log(var + self.eps)
        return res, torch.ones_like(z[:, 0]) * log_det.sum()
