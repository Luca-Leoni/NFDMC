import torch
import torch.nn as nn

from torch import Tensor

class Distribution(nn.Module):
    """
    Archetipe of a probability distribution inside the model
    """
    def __init__(self) -> None:
        super().__init__()


    def forward(self, num_sample: int = 1) -> tuple[Tensor, Tensor]:
        """
        Override of the torch.nn.Module forward method

        Needs to generate a certain number of sample from the distribution and compute also the log probability of obtaining it.

        Parameters
        ----------
        num_sample 
            number of samples wanted

        Returns
        -------
        tuple[Tensor, Tensor]
            Contains the samples in the first tensor and the log probability of every sample in the second

        Raises
        ------
        Not implemented:
            If is not implemented in the costumazied distribution it fails
        """
        raise NotImplementedError

    def log_prob(self, z: Tensor) -> Tensor:
        """
        Computes the log probability of the samples passed

        Parameters
        ----------
        z
            Tensor with dimension (batch, dim_sample) containing the different samples under analysis

        Returns
        -------
        Tensor
            Tensor containing the log probabilities of the samples

        Raises
        ------
        Not implemented:
            If is not implemented in the costumazied distribution it fails
        """
        raise NotImplementedError

    def sample(self, num_sample: int = 1) -> Tensor:
        """
        Sample from the distribution

        Parameters
        ----------
        num_sample
            number of samples to be drawn

        Returns
        -------
        Tensor
            Final samples

        Raises
        ------
        Not implemented:
            If is not implemented in the costumazied distribution it fails
        """
        raise NotImplementedError


class RSDistribution(Distribution):
    """
    Version of the distribution primitive that automatically implements the sample method by using the rejection sampling algorithm
    """
    def __init__(self, n_dim: int, prop_scale: Tensor = torch.tensor(1.0), prop_shift: Tensor = torch.tensor(0.0)):
        super().__init__()

        self._n_dim = n_dim
        self.register_buffer("_prop_scale", prop_scale)
        self.register_buffer("_prop_shift", prop_shift)

    def sample(self, num_sample: int = 1) -> Tensor:
        samples = torch.zeros(0, self._n_dim, device=self._prop_scale.device) # pyright: ignore

        while samples.shape[0] < num_sample:
            samples = torch.cat([self._rejection_sampling(num_sample), samples], dim=0)

        return samples[:num_sample,:]

    def _rejection_sampling(self, n_steps: int) -> Tensor:
        proposal = self._prop_scale * torch.rand(n_steps, self._n_dim, device=self._prop_scale.device) + self._prop_shift # pyright: ignore

        accept = torch.exp(self.log_prob(proposal)) < torch.rand(n_steps, device=self._prop_scale.device) # pyright: ignore

        return proposal[accept, :]
