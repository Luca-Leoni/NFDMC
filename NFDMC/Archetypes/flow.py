import torch.nn as nn

from torch import Tensor

class Flow(nn.Module):
    """
    Archetipe that defines a flow inside the model
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]: # pyright: ignore
        """
        Override of the torch.nn.Module function

        Transforms the input samples and returns the transformed results along with the log|det J(z)|

        Parameters
        ----------
        z
            Input batch of samples

        Returns
        -------
        tuple[Tensor, Tensor]
            Tuple with transformed z and with a tensor containing all the log det of the samples

        Raises
        ------
        Not implemented:
            If is not implemented in the costumazied distribution it fails
        """
        raise NotImplementedError

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]: # pyright: ignore
        """
        inverse of the forward transformation

        Parameters
        ----------
        z
            Batch of transformed samples that needs to get transformed back

        Returns
        -------
        tuple[Tensor, Tensor]
            untrasformed samples along with the log determinant of J(T(z))

        Raises
        ------
        Not implemented:
            If is not implemented in the costumazied distribution it fails
        """
        raise NotImplementedError
