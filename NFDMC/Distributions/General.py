import torch
import torch.nn as nn

from torch import Tensor
from .distribution import Distribution

class MultiGaussian(Distribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix for generating a D dimensional vector with entries composed of independent Gaussian variables.
    """
    def __init__(self, dimension: int, trainable: bool = True):
        """
        Constructor

        Parameters
        ----------
        dimension
            Dimension of the sample to draw
        trainable
            Sad if the means and the deviations of the Gaussians should be seen as parameters
        device
            Tell if you want to run on cpu or gpu
        """
        super().__init__()

        self._dim    = dimension

        if trainable:
            self._mean    = nn.Parameter(torch.zeros(dimension))
            self._std_dev = nn.Parameter(torch.ones(dimension))
        else:
            self.register_buffer("_mean", torch.zeros(dimension))
            self.register_buffer("_std_dev", torch.ones(dimension))


    def forward(self, num_sample: int = 1) -> tuple[Tensor, Tensor]:
        r"""
        Overload of the torch.nn.Module method

        Generates a selected number of sample drawn from the Multivariate Gaussian distribution and their log probabilities given as:
            .. math::
                p(\mathbb{z}) = -\frac{1}{2}\log(2\pi\prod_{i=1}^D \sigma_i^2) - \frac{1}{2}\sum_{i=1}^D \left(\frac{z_i - \mu_i}{\sigma_i}\right)^2

        Parameters
        ----------
        num_sample
            Number of sample to draw from the distribution

        Returns
        -------
        tuple[Tensor, Tensor]
            Tuple with batch of samples in the first tensor and log probabilities in the second
        """
        if self._mean is None or self._std_dev is None:
            raise TypeError()

        mean    = self._mean.expand(num_sample, self._dim)
        std_dev = self._std_dev.expand(num_sample, self._dim)

        z = torch.normal(mean=mean, std=std_dev)

        return z, - (torch.log(2*torch.pi*torch.prod(self._std_dev**2)) + torch.sum(torch.pow((z - mean)/std_dev, 2), 1))/2

    def log_prob(self, z: Tensor) -> Tensor:
        """
        Computes log probabilities of a batch of sample, to see the formula used look at forward method

        Parameters
        ----------
        z
            Batch of samples

        Returns
        -------
        Tensor
            Tensor with the log probabilities of the samples
        """
        if self._std_dev is None:
            raise TypeError()

        return - (torch.log(2*torch.pi*torch.prod(self._std_dev**2)) + torch.sum(torch.pow((z - self._mean)/self._std_dev, 2), 1))/2

    def sample(self, num_sample: int = 1) -> Tensor:
        """ 
        Sample from the distribution a certain batch

        Parameters
        ----------
        num_sample
            Number of samples wanted defining the batch

        Returns
        -------
        Tensor
            Batch with the samples
        """
        if self._mean is None or self._std_dev is None:
            raise TypeError()

        mean    = self._mean.expand(num_sample, self._dim) 
        std_dev = self._std_dev.expand(num_sample, self._dim)
        
        return torch.normal(mean=mean, std=std_dev)




class TwoMoon(Distribution):
    """
    Two Moon distribution generally used as a toy model to test architectures
    """
    def __init__(self) -> None:
        """
        Constructor
        """
        super().__init__()

    def log_prob(self, z: Tensor) -> Tensor:
        r"""
        Evaluates the log probability of a batch of samples evaluated using the equation:
            .. math::
                \log(p(\mathbb{z})) = - \frac{1}{2}\left(\frac{\norm{z} - 2}{0.2}\right)^2 + \log\left\{ \exp\left[ -\frac{1}{2}\left(\frac{z_0 - 2}{0.3}\right)^2 \right] + \exp\left[ -\frac{1}{2}\left(\frac{z_0 + 2}{0.3}\right)^2 \right] \right\}


        Parameters
        ----------
        z
            Batch with the samples

        Returns
        -------
        Tensor
            Tensor with the log probabilities of the batch
        """
        a = torch.abs(z[:,0])
        return -0.5 * ((torch.norm(z, dim=1) - 2) / 0.2) ** 2 - 0.5 * ((a - 2) / 0.3) ** 2 + torch.log(1 + torch.exp(-4 * a / 0.09))
