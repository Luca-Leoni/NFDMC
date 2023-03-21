import torch

from torch import Tensor
from ..Archetypes import Flow, DiaModule

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


class PermuteTimeBlock(Flow, DiaModule):
    def __init__(self, block: int) -> None:
        super().__init__()

        self.__n_couple = (self.get_dia_comp()[block][1] - self.get_dia_comp()[block][0]) // 2

        self.__permutation = torch.randperm(self.__n_couple)
        self.__inverse     = torch.argsort(self.__permutation)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        time_pos = self.get_dia_comp()[1]

        times = z[:, time_pos[0]:time_pos[1]].reshape(self.__n_couple, 2)

        z[:, time_pos[0]:time_pos[1]] = times[self.__permutation, :].flatten()

        return z, torch.zeros(z.shape[0], device=z.device)

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        time_pos = self.get_dia_comp()[1]

        times = z[:, time_pos[0]:time_pos[1]].reshape(self.__n_couple, 2)

        z[:, time_pos[0]:time_pos[1]] = times[self.__inverse, :].flatten()

        return z, torch.zeros(z.shape[0], device=z.device)


class SwapDiaBlock(Flow, DiaModule):
    def __init__(self, block1: int, block2: int) -> None:
        super().__init__()

        self.__b1 = block1
        self.__b2 = block2


    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        return z, torch.zeros(z.shape[0], device=z.device)

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        return z, torch.zeros(z.shape[0], device=z.device)
