import torch
import torch.nn as nn

from torch import Tensor
from .Archetypes import Flow, Distribution

class Manager(nn.Module):
    """
    Flow manager used to compose the different flows and perform the computations
    """
    def __init__(self, base: Distribution, flows: list[Flow], target: Distribution | None = None):
        """
        Constructor

        Parameters
        ----------
        base
            Base distribution used to sample things out
        flows
            List of the flows used to model the transformation
        target
            Target distribution that you want to approximate
        """
        super().__init__()

        self._base = base
        self._flows = nn.ModuleList(flows)
        self._target = target

    def forward(self, z: Tensor) -> Tensor:
        """
        Override of the torch.nn.Module method

        Apply the transformation on a batch of samples

        Parameters
        ----------
        z
            Batch with all the samples that you want to transform

        Returns
        -------
        Tensor
            Batch with all the transformed samples
        """
        for flow in self._flows:
            z, _ = flow(z)
        return z

    def sample(self, num_sample: int = 1) -> tuple[Tensor, Tensor]:
        r"""
        Sample from the approximated distribution giving the batch of wanted samples and the log probability

        The sample and probabilities are computed using the following known formulas to sample from a flow composed of $N$ transformations:
        .. math::
            \hat{\mathbb{z}} = T_N \circ \dots \circ T_1 (\mathbb{z}), \hspace{2cm} \hat{p}(\hat{\mathbb{z}}) = p(\mathbb{z})\prod_{n=1}^N \abs{\det J_{T_k}(\mathbb{z}_k)}^-1
        where $\mathbb{z}_k$ is the output of the $k$-th transformation

        Parameters
        ----------
        num_sample
            number of samples wanted

        Returns
        -------
        tuple[Tensor, Tensor]
            tuple containing the batch with all the samples and a tensor with the final log probabilities
        """
        z, log_p = self._base(num_sample)
        for flow in self._flows:
            z, log_det = flow(z)
            log_p -= log_det
        return z, log_p

    def log_prob(self, z: Tensor) -> Tensor:
        """
        Compute the log probability of the model

        Parameters
        ----------
        z
            Batch of samples to compute the log probability from

        Returns
        -------
        Tensor
            Log probabilities of the batch of samples
        """
        log_p = torch.zeros(len(z), dtype=z.dtype, device=z.device)
        x = z
        for i in range(len(self._flows) - 1, -1, -1):
            x, log_det = self._flows[i].inverse(x) # pyright: ignore
            log_p += log_det
        log_p += self._base.log_prob(x)
        return log_p

    def forward_kdl(self, z: Tensor) -> Tensor:
        r"""
        Compute the forward KDL divergence of the model

         The forward KDL is a distance inside the space of probability distribution that can be approximated via Monte Carlo sampling as follows:
        .. math::
            L(\mathbb{\theta}) \approx -\frac{1}{M} \sum_{m=1}^M \log q(T^{-1}(\mathbb{z}_m)) + \log \abs{\det J_{T^{-1}}(\mathbb{z}_m)}

        Parameters
        ----------
        z
            Batch with the samples from the target distribution

        Returns
        -------
        Tensor
            Forward KDL loss of the model
        """
        log_p = torch.zeros(z.shape[0], device=z.device)
        x = z
        for i in range(len(self._flows) - 1, -1, -1):
            x, log_det = self._flows[i].inverse(x) # pyright: ignore
            log_p += log_det
        log_p += self._base.log_prob(x)
        return -torch.mean(log_p)

    def reverse_kdl(self, num_sample: int) -> Tensor:
        r"""
        Compute the reverse KDL divergence of the model

        The reverse KDL is a distance inside the space of probability distribution that can be approximated via Monte Carlo sampling as follows:
        .. math::
            L(\mathbb{\theta}) \approx \frac{1}{M} \sum_{m=1}^M \log p(\hat{\mathbb{z}}_m) - \log \hat{p}(\hat{\mathbb{z}}_m)


        Parameters
        ----------
        num_sample
            number of sample to use inside the evaluation of the loss, the more the better.

        Returns
        -------
        Tensor
            Reverse KDL loss

        Raises
        ------
        ValueError:
            If the target distribution is not given the target log probability can't be estimated
        """
        if isinstance(self._target, type(None)):
            raise ValueError("The target distribution needs to be defined to perform reverse KLD measure!")

        z, mlog_p = self._base(num_sample)

        bad = (z[:, 0] > 25) | (z[:, 0] <= 0) | (z[:, 1] <= 0) | (z[:, 1] > 50)
        if bad.any():
            print(z[bad])
            raise RuntimeError("Generation fucked up!")

        for i, flow in enumerate(self._flows):
            z, log_det = flow(z)
            mlog_p -= log_det

            if (z[:, 0] > 25).any():
                print(z[z[:, 0] > 25])
                raise RuntimeError(f"Order overshoot from flow {i}!")

            if (z[:, 1] > 50).any():
                print(z[z[:,1] > 50])
                raise RuntimeError(f"Time of flight overshoot from flow {i}!")

            if (z[:, 2:] > z[:, 1:2]).any():
                print(z[(z[:, 2:] > z[:, 1:2]).any(dim=1)])
                raise RuntimeError(f"Phonon time overshoot from flow {i}!")
            
            bad = (z[:, 2::2] > z[:, 3::2]).any(dim=1)
            if bad.any() and i > 6:
                print(z[bad])
                raise RuntimeError(f"Phonon time unordered from flow {i}!")

            bad = torch.isnan(log_det) | torch.isinf(log_det)
            if bad.any():
                print(z[bad])
                raise RuntimeError(f"Log det exploded from flow {i}!")
            
            bad = torch.isnan(mlog_p) | torch.isinf(mlog_p)
            if bad.any():
                print(z[bad])
                print(log_det[bad])
                raise RuntimeError(f"Model prob exploded from flow {i}!")


        tlog_p = self._target.log_prob(z)

        return - torch.mean(tlog_p) + torch.mean(mlog_p)
