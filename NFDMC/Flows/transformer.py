import torch

from ..Archetypes import Transformer
from torch import Tensor

#-------------------------------

class Affine(Transformer):
    """
    Definition of the affine transformer as a basic linear transformation
    """
    def __init__(self):
        """
        Constructor

        Defines the affine coupling transformation as:
            .. math::
                z'_i = exp(h_0^i) * z_i + h_1^i
        So that both log det and inverse are analitically known, also the number of parameters needed for variable dimension is 2.
        """
        super().__init__(2)

    def forward(self, z: Tensor, h: Tensor) -> Tensor:
        """
        Override of the torch.nn.Module method

        Parameters
        ----------
        z
            Input variable
        h
            Parameters of the transformation

        Returns
        -------
        Tensor
            Output variable
        """
        return torch.exp(h[:, ::2]) * z + h[:, 1::2]

    def inverse(self, z: Tensor, h: Tensor) -> Tensor:
        """
        Inverse transformation evaluated through:
            .. math::
                z_i = (z_i' - h_1^i) * exp(-h_0^i)     

        Parameters
        ----------
        z
            Transformed variable
        h
            Parameters that defines the transformation

        Returns
        -------
        Tensor
            Untransformed variable
        """
        return (z - h[:, 1::2]) * torch.exp(-h[:, ::2])

    def log_det(self, _: Tensor, h: Tensor) -> Tensor:
        r"""
        Compute the log determinant of the Jacobian which ends up in being simply:
            .. math::
                \log\det J = \sum_i h_0^i

        Parameters
        ----------
        _
            Input variable, not use in this case
        h
            Parameter list evaluated using the input variable

        Returns
        -------
        Tensor
            Log determinant
        """
        return torch.sum(h[:, ::2], dim = 1)
