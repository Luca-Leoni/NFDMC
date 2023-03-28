import torch
import math

from torch import Tensor
from ..Archetypes import Distribution, RSDistribution, Diagrammatic

#---------------------------------------

class Holstein(Distribution, Diagrammatic):
    def __init__(self, on_site: float, phonon: float, coupling: float):
        r"""
        Construct the Holstein target distribution by passing the constants of the Toy Hamiltonian. In particular this distribution contains the log_probability evaluation by using the log_weight for it. Also, the diagrams are assumed to be represented, for now, as:
            $[order, \tau, \tau_1^c, \tau_1^d, \tau_1^c, \tau_2^d, \dots]$

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
                \log W = n\log g + \mu\tau + \sum_{i=1}^n \Omega (\tau_i^c - tau_i^d)    

        Parameters
        ----------
        z
            Batch with diagrams

        Returns
        -------
        Tensor
            log probabilities of the batch
        """
        dia_con = self.get_dia_comp()
        beg_ph = dia_con[2, 0]
        end_ph = dia_con[2, 1]

        # If time ordering or positiveness is not respected give really low probability
        # ordered = (z[:, beg_ph:end_ph:2] < z[:, beg_ph+1:end_ph:2]).all(dim=1)
        # positive = (z >= 0).all(dim=1)

        # if (ordered != positive).all():
        #     print(z)

        # Tranform first element in order of diagram
        order = torch.floor(z[:,dia_con[0,0]:dia_con[0,1]])*2

        # Set to zero the element out of the order
        zeros = torch.arange(z.shape[1] - 2, device=z.device) >= order
        log_weight = torch.clone(z[:, beg_ph:end_ph])
        log_weight[zeros] = 0


        # Compute the weight
        log_weight =  (self.__g * order + self.__mu * z[:, dia_con[1,0]:dia_con[1,1]]).flatten() +  self.__om * torch.sum((log_weight[:, ::2] - log_weight[:, 1::2]), dim=1)

        # select right output
        return log_weight
        # return torch.where(ordered == positive, log_weight, -1000000 * torch.abs(log_weight)) 
