import torch
import math
import torch.nn as nn

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
        # Get time of flight
        tm_fly = self.get_block_from("tm_fly", z)

        # Tranform first element in order of diagram
        order = torch.floor(self.get_block_from("order", z))*2

        # Set to zero the element out of the order
        zeros = torch.arange(z.shape[1] - 2, device=z.device) >= order
        log_weight = torch.clone(self.get_block_from("phonons", z))
        log_weight[zeros] = 0

        # Compute the weight
        log_weight =  (self.__g * order - self.__mu * tm_fly).flatten() +  self.__om * torch.sum((log_weight[:, ::2] - log_weight[:, 1::2]), dim=1)

        # select right output
        return log_weight



class BaseHolstein(Distribution, Diagrammatic):
    """
    Base distribution used for the single site Holstein model
    """
    def __init__(self, max_order: int, max_tm_fly: float = 50., trainable: bool = False, rateo: Tensor = torch.tensor(1.0)) -> None:
        r"""
        Constructor

        Creates a base distribution for the single site Holstein model, where the three main type of variables are drown in the following way:
            - order ~ $U(0,$ max_order)
            - tm_fly ~ $E(\tau; rateo) = rateo e^{-rateo\tau}$
            - phonon ~ $U(0,$ tm_fly)

        Parameters
        ----------
        max_order
            Maximum order possible for the diagram, will also define the dimension of the array representing the diagram
        trainable
            Tells if the rateo can be trained like a parameter
        rateo
            Initial value of the decaing rate in the exponential distribution of the time of flight
        """
        super().__init__()

        self.__max_or = max_order
        self.__max_tm = max_tm_fly

        if trainable:
            self.rateo_tm = nn.Parameter(rateo)
            self.rateo_or = nn.Parameter(rateo)
        else:
            self.register_buffer("rateo_tm", rateo)
            self.register_buffer("rateo_or", rateo)


    def forward(self, num_sample: int = 1) -> tuple[Tensor, Tensor]:
        """
        Override of the torch.nn.Module method

        Sample a wanted number of diagrams from the distribution and returns the log probability of them

        Parameters
        ----------
        num_sample
            Number of diagrams to draw

        Returns
        -------
        tuple[Tensor, Tensor]
            Batch of diagrams and log probability of them
        """
        samples = self.sample(num_sample)
        return samples, self.log_prob(samples)


    def sample(self, num_sample: int = 1) -> Tensor:
        """
        Sample from the distribution as described in the constructor.
        
        The order of the diagram is generated between 0 and half of the max_order since in the target distribution such value is multiplied by 2 in order to have an even order since the model requires it.

        Parameters
        ----------
        num_sample
            Number of diagrams to draw

        Returns
        -------
        Tensor
            Batch of diagrams
        """
        R = torch.rand(size=(num_sample, 2), device=self.rateo_tm.device) * 0.9999999
        order  = -torch.log(0.9999999 - R[:, 0:1] * (1 - torch.exp(-self.rateo_or * self.__max_or * 0.5))) / self.rateo_or
        tm_fly = -torch.log(0.9999999 - R[:, 1:2] * (1 - torch.exp(-self.rateo_tm * self.__max_tm))) / self.rateo_tm
        couple = torch.rand(size=(num_sample, self.__max_or), device=self.rateo_tm.device) * tm_fly

        return torch.cat((order, tm_fly, couple), dim=1)


    def log_prob(self, z: Tensor) -> Tensor:
        r"""
        Computes the log probability of the batch of diagrams given
        .. math::
            \log p = -\log(max_order/2) + \log(rateo) - rateo\tau - max_order\log(\tau)

        Parameters
        ----------
        z
            Batch of diagrams

        Returns
        -------
        Tensor
            Log probability of every diagram
        """
        tm_fly = self.get_block_from("tm_fly", z).flatten()
        order  = self.get_block_from("order", z).flatten()

        return torch.log(self.rateo_tm * self.rateo_or) - torch.log(1 - torch.exp(-self.rateo_tm * self.__max_tm)) - torch.log(1 - torch.exp(-self.rateo_or * self.__max_or * 0.5)) - self.rateo_tm * tm_fly - self.rateo_or * order - self.__max_or * torch.log(tm_fly)


class SBaseHolstein(Distribution):
    def __init__(self, tm_fly: float, max_order: int) -> None:
        super().__init__()

        self.__tm_fly = torch.tensor(tm_fly)
        self.__max    = max_order


    def forward(self, num_sample: int = 1) -> tuple[Tensor, Tensor]:
        samples = self.sample(num_sample)
        return samples, self.log_prob(samples)


    def sample(self, num_sample: int = 1) -> Tensor:
        order  = -torch.log(torch.rand(num_sample, 1, device=self.__tm_fly.device))
        phonon = torch.rand(num_sample, self.__max, device=self.__tm_fly.device) * self.__tm_fly
        return torch.cat( (order, phonon), dim=1 )


    def log_prob(self, z: Tensor) -> Tensor:
        return -z[:, 0] - self.__max * torch.log(self.__tm_fly)


class SHolstein(Distribution):
    def __init__(self, phonon: float, coupling: float) -> None:
        super().__init__()

        self.__Omega = phonon
        self.__g = math.log(coupling)


    def log_prob(self, z: Tensor) -> Tensor:
        order  = torch.floor(z[:, 0:1])
        phonon = z[:, 1:]

        zeros  = torch.arange(phonon.shape[1], device=z.device).expand(*phonon.shape) > order
        phonon[zeros] = 0

        return self.__g * order * 2 - self.__Omega * torch.sum(phonon, dim=1)


