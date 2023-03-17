import torch

from ..Archetypes import Flow, Transformer, Conditioner
from torch import Tensor

#------------------------------

class Autoregressive(Flow):
    """
    Defines an autoregressive flow model
    """
    def __init__(self, trans: Transformer, cond: Conditioner):
        r"""
        Constructor

        Defines an autoregressive flow given by the transformation defined in the transformer depending on the parameters computed trough the conditioner passed. It's important that in this model the inverse of the transformer is defined so that it can be performed element wise since the autoregressive flow model can only work in that way.

        Parameters
        ----------
        trans
            Transformer giving out the form of $\tau$
        cond
            Conditioner that tells how to compute the parameters $\mathbb{h}$ of the transformer

        Raises
        ------
        ValueError:
            If the number of parameters generated per variable dimension by the conditioner are not the one expected by the selected transformer
        """
        super().__init__()

        if trans.trans_features != cond.trans_features:
            raise ValueError("Conditioner trans_features doesn't match the wanted trasnsformer one!")

        self.__trans = trans
        self.__cond  = cond

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Override of the torch.nn.Module method

        Perform the evaluation of the transformation by assuming that the conditioner is able to evaluate all the parameters at once, no recurrent architecture.

        Parameters
        ----------
        z
            Input variable

        Returns
        -------
        tuple[Tensor, Tensor]
            Transformed variable and log determinant of the transformation
        """
        h = self.__cond(z)
        return (self.__trans(z, h), self.__trans.log_det(z, h))

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute the inverse of the trasnformation by doing an element wise invertion of the variable. In particular it passes to the inverse trasnformation of the transformer an array with $z_i$ of the batch and $\mathbb{h_i}$ of the batch

        Parameters
        ----------
        z
            Transformed variable

        Returns
        -------
        tuple[Tensor, Tensor]
            Untransformed variable and log determinant of the inverse transformation
        """
        h = self.__cond(z)

        for i in range(z.shape[1]):
            z[:, i] = self.__trans.inverse(z[:, i], h[:, self.__trans.trans_features * i: self.__trans.trans_features * i + 2])
            h = self.__cond(z)

        return z, -self.__trans.log_det(z, h)
