import torch

from torch import Tensor
from ..Archetypes import Flow

class PermuteRandom(Flow):
    """
    Flows that randomly permutes the elements inside the input variables
    """
    def __init__(self, var_dim: int):
        """
        Constructor

        Parameters
        ----------
        var_dim
            Dimensions of the input variable
        """
        super().__init__()

        self.register_buffer("_per_for", torch.randperm(var_dim))
        self.register_buffer("_per_inv", torch.argsort(self._per_for)) #pyright: ignore


    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Override of the torch.nn.Module method

        Parameters
        ----------
        z
            Input data

        Returns
        -------
        tuple[Tensor, Tensor]
            Output data and log det of the transformation, in this case is 0
        """
        return z[:, self._per_for], torch.zeros(z.shape[0], device=z.device)

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Inverse of the transformation

        Parameters
        ----------
        z
            Transformed input variable

        Returns
        -------
        tuple[Tensor, Tensor]
            Untransformerd variable and log det of the inverse, 0 also in this case
        """
        return z[:, self._per_inv], torch.zeros(z.shape[0], device=z.device)


class Swap(Flow):
    def __init__(self):
        super().__init__()

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        return torch.flip(z, dims=(1,)), torch.zeros(z.shape[0], device=z.device)

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        return torch.flip(z, dims=(1,)), torch.zeros(z.shape[0], device=z.device)
