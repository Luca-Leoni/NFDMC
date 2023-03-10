import torch
import numpy as np

from torch import Tensor
from . import Distribution

class Base(Distribution):
    """
    Base simple distribution to create Holstein diagrams with a simple distribution.

    The diagrams are seen as simple arrays with a lenght given by max_order + 1 where the first entry is an integer giving the order, while the others are couples [t', t''] where t' defines the creation of a phonon and t'' its destruction.
    """
    def __init__(self, tm_fly: float, max_order: int = 100):
        """
        Constructor

        Parameters
        ----------
        tm_fly
            Time of flight of the electron studied
        max_order
            Maximum order used to represente the diagrams

        Raises
        ------
        ValueError:
            The maximum order needs to be even since the diagram in the model have only even order
        """
        super().__init__()

        if max_order % 2 == 1:
            raise ValueError("The maximum order needs to be an even number!")

        self.register_buffer("tm_fly", torch.full(size=(1,), fill_value=tm_fly))
        self.register_buffer("max_order", torch.full(size=(1,), fill_value=max_order))


    def forward(self, num_sample: int = 1) -> tuple[Tensor, Tensor]:
        """
        Override of the torch.nn.Module function

        Generates the wanted number of digrams along with their log probability

        Parameters
        ----------
        num_sample
            Number of diagrams to draw

        Returns
        -------
        tuple[Tensor, Tensor]
            Batch with diagrams sampled and array with relatives log probabilities

        Raises
        ------
        ValueError:
            If the tm_fly and max_order are not initialized properly
        """
        if self._buffers['tm_fly'] is None or self._buffers['max_order'] is None:
            raise ValueError("Buffers have not been initialized correctly")

        tm_fly    = float(self._buffers['tm_fly'].cpu().numpy())
        max_order = int(self._buffers['max_order'].cpu().numpy())
        device    = self._buffers['tm_fly'].device

        # Initialize the diagram
        dia = torch.zeros(num_sample, max_order+1, dtype=torch.float, device=device)

        # Generate the diagrams orders
        dia[:,0] = torch.randint(low=0, high=max_order+1, size=(num_sample,)) // 2

        # Generate the couples 
        couples = torch.rand(num_sample, max_order // 2, 2, device=device)
        couples[:,:,0] *= tm_fly
        couples[:,:,1]  = couples[:,:,0] + (tm_fly - couples[:,:,0])*couples[:,:,1]

        # Flatten out and put things in place
        dia[:, 1:] = couples.flatten(1)[:,0:]

        # Compute the log probability
        log_p = torch.log(torch.prod(tm_fly * (tm_fly - dia[:,1::2]), 1))

        return dia, - np.log(max_order + 1) - log_p


    def log_prob(self, z: Tensor) -> Tensor:
        """
        Override of the primitive method

        Parameters
        ----------
        z
            Batch of samples that we want to compute the log probability of

        Returns
        -------
        Tensor
            Log probabilities of samples

        Raises
        ------
        ValueError:
            If the tm_fly and max_order are not initialized correctly
        """
        if self._buffers['max_order'] is None or self._buffers['tm_fly'] is None: 
            raise ValueError("Buffers have not been initialized correctly")

        return - torch.log(self._buffers['max_order'] + 1) - torch.log(torch.prod(self._buffers['tm_fly'] * (self._buffers['tm_fly'] - z[:,1::2]), 1))
