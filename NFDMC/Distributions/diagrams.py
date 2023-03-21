import torch
import math

from torch import Tensor
from ..Archetypes import Distribution, RSDistribution

class Holstein(Distribution):
    def __init__(self, on_site: float, phonon: float, coupling: float):
        r"""
        Construct the Holstein target distribution by passing the constants of the Toy Hamiltonian. In particular this distribution contains the log_probability evaluation by using the log_weight for it. Also, the diagrams are assumed to be represented, for now, as:
            $[order, \tau_1^c, \tau_1^d, \tau_1^c, \aut_2^d, \dots]$

        Parameters
        ----------
        on_site
            $\mu$ value of the Hamiltonian, electron energy
        phonon
            $\Omega$ value of the Hamiltonian, phonon energy
        coupling
            $g$ value of the Hamiltonian, coupling energy
        """
        super().__init__()

        self.__mu  = on_site
        self.__om  = phonon
        self.__g   = math.log(coupling)

    def log_prob(self, z: Tensor) -> Tensor:
        r"""
        Computes the log probability as the log weight of the diagram, therefore the formula is:
            .. math::
                \log W = n\log g + \sum_{i=1}^n \Omega (\tau_i^c - tau_i^d)    

        Parameters
        ----------
        z
            Batch with diagrams

        Returns
        -------
        Tensor
            log probabilities of the batch
        """
        # If time ordering is not respected give really low probability
        ordered = (z[:, 2::2] > z[:, 1::2]).all(dim=1)

        # Tranform first element in order of diagram
        order = torch.floor(z[:,0]).reshape(z.shape[0], 1)

        # Set to zero the element out of the order
        zeros = torch.arange(z.shape[1] - 1).expand(z.shape[0], z.shape[1] - 1) > order
        log_weight = torch.clone(z[:, 1:])
        log_weight[zeros] = 0

        # Compute the weight
        log_weight =  self.__g * order.flatten() +  torch.sum((log_weight[:, ::2] - log_weight[:, 1::2]) * self.__om, dim=1)

        # select right output
        return torch.where(ordered, log_weight, -1000000) 
