import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional
from .Flows.flow import Flow
from .Distributions.distribution import Distribution

class Manager(nn.Module):
    """
    Flow manager used to compose the different flows and perform the computations
    """
    def __init__(self, base: Distribution, flows: list[Flow], target: Optional[Distribution|None] = None):
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

    def reverse_kld(self, num_sample: int):
        r"""
        Compute the reverse KDL divergence of the model

        The reverse KDL is a distance inside the space of probability distribution that can be approximated via Monte Carlo sampling as follows:
        .. math::
            L(\mathbb{\theta}) \approx \frac{1}{M} \sum_{m=1}^M \log p(\hat{\mathbb{z}}_m) - \log \hat{p}(\hat{\mathbb{z}}_m)


        Parameters
        ----------
        num_sample
            number of sample to use inside the evaluation of the loss, the more the better.

        Raises
        ------
        ValueError:
            If the target distribution is not given the target log probability can't be estimated
        """
        if self._target is None:
            raise ValueError("The target distribution needs to be defined to perform reverse KLD measure!")

        z, mlog_p = self._base(num_sample)
        for flow in self._flows:
            z, log_det = flow(z)
            mlog_p -= log_det

        tlog_p = self._target.log_prob(z)

        return torch.mean(tlog_p) - torch.mean(mlog_p)
