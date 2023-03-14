import torch

from ..flow import Flow
from torch import Tensor
from torch.nn import Module

class Affine(Flow):
    """
    Affine transformer in the most basic implementation possible
    """
    def __init__(self, conditioner: Module):
        r"""
        Constructor

        Takes as input a conditioner that creates the set of parameters to use inside the transformer that in this case are two parameter per random variable having that the transformation used is:
            .. math::
                z'_i = \exp(\alpha_i) * z_i + \beta_i
        so that the output of the conditioner needs to have the form $[\alpha_1, \beta_1, \alpha_2, \beta_2, \dots]$

        Parameters
        ----------
        conditioner
            Conditioner to use in the computations

        Raises
        ------
        ValueError:
            A control is done on the variable _trans_features to see if it's equal to 2
        """
        super().__init__()
        
        if conditioner._trans_features != 2:
            raise ValueError()

        self._cond = conditioner


    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Override of the torch.nn.Module method

        Parameters
        ----------
        z
            Input variable

        Returns
        -------
        tuple[Tensor, Tensor]
            Transformed variable along with the log determinant of the Jacobian of the transformation
        """
        h = self._cond(z)
       
        return torch.exp(h[:, ::2]) * z + h[:, 1::2], torch.sum(h[:, ::2], 1)


    def inverse(self, z1: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Compute the inverse of the transformation using the analitical formula, which unfortunatly is recursive in a general way. In particular the transformation is:
            .. math::
                z_i = \frac{z'_i - \beta_i}{\exp(\alpha_i)}, \hspace{2cm} {\alpha_i, \beta_i} = c_i(z_{<i})

        Parameters
        ----------
        z1
            Transformed variable

        Returns
        -------
        tuple[Tensor, Tensor]
            Untransformed variabel along with the log determinant of the inverse transformation
        """ 
        h = self._cond(z1)
        z = torch.clone(z1)
        z[:, 0] = (z[:,0] - h[0,1])/torch.exp(h[0,0])

        for i in range(1, z1.shape[1]):
            h = self._cond(z)
            z[:,i] = (z[:,i] - h[:, 2*i+1])/torch.exp(h[:, 2*i])

        return z, -torch.sum(self._cond(z)[:, ::2], dim=1)
