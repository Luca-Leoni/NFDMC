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
