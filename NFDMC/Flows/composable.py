import torch

from ..Archetypes import Flow, Transformer, Conditioner
from torch import Tensor
from torch.nn import Module

#------------------------------

class Autoregressive(Flow):
    """
    Defines an autoregressive flow model
    """
    def __init__(self, trans: Transformer, cond: Conditioner):
        r"""
        Constructor

        Defines an autoregressive flow given by the transformation defined in the transformer depending on the parameters computed trough the conditioner passed. It's important that in this model the inverse of the transformer is defined so that it can be performed element wise.

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
        r"""
        Compute the inverse of the transformation by doing an element wise invertion of the variable. In particular it passes to the inverse trasnformation of the transformer an array with $[[z_i^1], [z_i^2], \dots]$ of the batch and $[[h_{1,i}^1, h_{1,i}^2], \dots]$ of the batch

        Parameters
        ----------
        z
            Transformed variable

        Returns
        -------
        tuple[Tensor, Tensor]
            Untransformed variable and log determinant of the inverse transformation
        """
        h = self.__cond(z) # TODO: find a way to avoid those cloning

        for i in range(z.shape[1]):
            z[:, i] = self.__trans.inverse(z[:,i].reshape(z.shape[0], 1), 
                                           h[:, self.__trans.trans_features * i: self.__trans.trans_features * i + 2]).flatten()
            h = self.__cond(z)

        return z, -self.__trans.log_det(z, h)




class Coupling(Flow):
    """
    Defines a coupling layer where only a part of the input is transformed, and the parameters are computed on the other first part
    """
    def __init__(self, trans: Transformer, cond: Module, split: list[int] | None = None):
        """
        Constructor

        Construct a Coupling flow using the transformation given depending on the parameters computed through a general neural network that work as conditioner in this case. In particular the input variable will be splitted in $[z_1, z_2]$ with dimension specified in split and then the following transformation is done:
            .. math::
                z_2' = \tau(z_2, c(z_1))
        so that the parameters are evaluated depending only on $z_1$ and the Jacobian will be lower triangular for sure and c can be whatever network. at the end the input is $[z_1, z_2']$.

        Parameters
        ----------
        trans
            Transformer that defines the $\tau$
        cond
            Network used to evaluate the parameters to use in the transformer computations
        split
            List containing the two dimensions of the slice in which the input variable will be splitted, if not inserted the input will be splitted in half
        """
        super().__init__()

        self.__trans = trans
        self.__cond  = cond

        if isinstance(split, type(None)):
            self.__split = lambda x: torch.split(x, x.shape[1] // 2, dim = 1)
        elif len(split) != 2:
            raise ValueError("The input variable needs to be splitted in two parts, different number of dimensions inserted!")
        else:
            self.__split = lambda x: torch.split(x, split, dim = 1)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Override of the torch.nn.Module method

        Evaluate the transformation computing also the log determinant of the jacobian

        Parameters
        ----------
        z
            Batch with input variable

        Returns
        -------
        tuple[Tensor, Tensor]
            Transformed variable with log determinant of the Jacobian
        """
        z1, z2 = self.__split(z)
        h = self.__cond(z1) 
        return torch.cat((z1, self.__trans(z2, h)), dim = 1), self.__trans.log_det(z2, h)

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Conput the inverse of the transformation in a really simple way since now the inverse can be computed directly without iterating as:
            .. math::
                z_2 = \tau^{-1}(z_2', c(z_1))

        Parameters
        ----------
        z
            Batch with tranformed input variables

        Returns
        -------
        tuple[Tensor, Tensor]
            Untransformed variables and log det Jacobian of the inverse
        """
        z1, z2 = self.__split(z)
        h = self.__cond(z1)
        z2 = self.__trans.inverse(z2,h)
        return torch.cat((z1, z2), dim = 1), -self.__trans.log_det(z2, h)
