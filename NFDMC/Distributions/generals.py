import torch
import torch.nn as nn

from torch import Tensor, dtype
from ..Archetypes import Distribution, RSDistribution

#---------------------------------

class MultiGaussian(Distribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix for generating a D dimensional vector with entries composed of independent Gaussian variables.
    """
    def __init__(self, dimension: int, trainable: bool = True, mean: float = 0, std_dev: float = 1):
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

        self.__dim    = dimension

        if trainable:
            self._mean    = nn.Parameter(torch.full((dimension,), mean, dtype=torch.float64))
            self._std_dev = nn.Parameter(torch.full((dimension,), std_dev, dtype=torch.float64))
        else:
            self.register_buffer("_mean", torch.full((dimension,), mean, dtype=torch.float64))
            self.register_buffer("_std_dev", torch.full((dimension,), std_dev, dtype=torch.float64))


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
        z = self.sample(num_sample)
        return z, self.log_prob(z)

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
                              # log(sqrt(2*pi))
        return - self.__dim * 0.9189385332046727 - torch.sum(torch.log(self._std_dev) + 0.5 * torch.pow((z - self._mean)/self._std_dev, 2), dim=1)

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
        z = torch.randn( (num_sample, self.__dim), dtype=self._mean.dtype, device=self._mean.device)
        return self._mean + z * self._std_dev


class MultiModalGaussian(Distribution):
    """
    Defines a Gaussian distribution composed of various multivariate gaussian distribution in the same space
    """
    def __init__(self, n_dim: int, n_mod: int, mean: Tensor | None = None, std_dev: Tensor | None = None, trainable: bool = True):
        r"""
        Constructor of the multimodal gaussian, takes as imputs the dimensions of the variable to generate and the number of modes wanted in the distribution.

        Parameters
        ----------
        n_dim
            Number of dimension of the random variable
        n_mod
            Number of multivariate Gaussians to take into account in the total one
        mean
            Possible initialization of the mean of the different gaussians as $[[\mu_1^1, \mu_1^2, \dots], [\mu_2^1, \dots], \dots]$ drawn randomly otherwise
        std_dev
            Possible initialization of the standard deviations equal to means if not inserted are initialized to ones
        trainable
            Tells if means and std_dev should be collexted as parameters
        """
        super().__init__()

        self._n_dim = n_dim

        if isinstance(mean, type(None)):
            mean = 4*torch.rand(n_mod, n_dim) - 2

        if isinstance(std_dev, type(None)):
            std_dev = torch.ones(n_mod, n_dim)

        if trainable:
            self.mean = nn.Parameter(mean)
            self.std_dev = nn.Parameter(std_dev)
        else:
            self.register_buffer("mean", mean)
            self.register_buffer("std_dev", std_dev)

    def forward(self, n_sample: int) -> tuple[Tensor, Tensor]:
        """
        Override of the torch.nn.Module methos

        Draw a certain number of sample from the distribution and returns also the log probaility of them

        Parameters
        ----------
        n_sample
            Number of samples

        Returns
        -------
        tuple[Tensor, Tensor]
            Samples drawn and log probability
        """
        z = self.sample(n_sample)
        return z, self.log_prob(z)

    def sample(self, num_sample: int = 1) -> Tensor:
        """
        Samples from the distribution

        Parameters
        ----------
        num_sample
            Number of samples to draw

        Returns
        -------
        Tensor
            Samples drawn
        """
        samples = torch.randn(num_sample, self._n_dim, dtype=self.mean.dtype, device=self.mean.device)
        mode    = torch.randint(low=0, high=self.mean.shape[0], size=(num_sample,), device=self.mean.device)

        samples = samples * self.std_dev[mode, :] + self.mean[mode, :]

        return samples

    def log_prob(self, z: Tensor) -> Tensor:
        """
        Computes the log probability of a Batch of samples

        Parameters
        ----------
        z
            Batch of samples

        Returns
        -------
        Tensor
            Log probabilities
        """
        
        log_p = torch.zeros(z.shape[0], self.mean.shape[0], dtype=z.dtype, device=z.device)

        for i in range(self.mean.shape[0]):
            log_p[:, i] += -torch.sum(torch.pow((z - self.mean[i])/self.std_dev[i], 2), dim=1)/2
                                              # log(sqrt(2 * pi))
        return torch.logsumexp(log_p, dim=1) - 0.9189385332046727 - torch.log(torch.sum(torch.prod(self.std_dev, dim=1)))


class MultiExponential(Distribution):
    def __init__(self, n_dim: int, trainable: bool = False, rateo: Tensor | float = 1.):
        r"""
        Constructor

        Generate a multi exponential distribution where every variable is drawn from a separate distribution given by:
            .. math::
                p(z_i) = \lambda_i e^{-\lambda_i z_i}
        so that every element is given inside $\mathbb{R}_+$ with a different rateo $\lambda_i$

        Parameters
        ----------
        n_dim
            Number of dimensions of the random variable
        trainable
            Tells if the $\lambda$ should be counted as paramters to train 
        rateo
            Starting values of the $\lambda$ to use for every variable
        """
        super().__init__()

        self.__n_dim = n_dim

        if isinstance(rateo, float):
            rateo = torch.full(size=(n_dim,), fill_value=rateo)
        elif rateo.shape[0] != n_dim:
            raise ValueError("The ratei tensor dimension don't match the selected one in exponential distribution!")

        if trainable:
            self.rateo = nn.Parameter(rateo.type(torch.float64))
        else:
            self.register_buffer("rateo", rateo.type(torch.float64))

    def forward(self, num_sample: int = 1) -> tuple[Tensor, Tensor]:
        """
        Override of the torch.nn.Module method

        Draws the wanted number of sample and returns them alogn withe their log probaility

        Parameters
        ----------
        num_sample
            Number of samples to draw

        Returns
        -------
        tuple[Tensor, Tensor]
            Batch with the samples and log probability of them
        """
        z = self.sample(num_sample)
        return z, self.log_prob(z)

    def sample(self, num_sample: int = 1) -> Tensor:
        """
        Draws samples from the distribution

        Parameters
        ----------
        num_sample
            Number of sample to draw

        Returns
        -------
        Tensor
            Batch with the sample
        """
        z = torch.rand(num_sample, self.__n_dim, device=self.rateo.device, dtype=torch.float64)
        return - (-z).log1p() / self.rateo

    def log_prob(self, z: Tensor) -> Tensor:
        """
        Computes the log probabilities of the batch of sample given

        Parameters
        ----------
        z
            Batch with the samples

        Returns
        -------
        Tensor
            Log probabilities of the samples
        """
        return self.rateo.log().sum() - torch.sum(self.rateo * z, dim=1) 


class TwoMoon(RSDistribution):
    """
    Two Moon distribution generally used as a toy model to test architectures
    """
    def __init__(self) -> None:
        """
        Constructor
        """
        super().__init__(2, prop_scale = torch.tensor(6.0), prop_shift = torch.tensor(-3.0))

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
