import torch
import torch.nn as nn

from torch.nn.utils import weight_norm
from torch import Tensor

class MLP(nn.Module):
    """
    General network composed of linear layers and leaky ReLU activation functions
    """
    def __init__(self, *dims: int, leaky: float = 0.0, bias: bool = True, activate_out: bool = False, init_zero: bool = False, weight_nor: bool = False):
        """
        Constructor

        Construct the network with the given dimensions

        Parameters
        ----------
        dims
            Dimensions of the network
        leaky
            Leaky parameters for the LeakyReLU activation functions
        bias
            Tells if the bias should be present inside the linear layers
        init_zero
            Tells if the last Linear layer should be initialized to zero
        """
        super().__init__()

        net = []
        for i in range(len(dims)-1):
            if i == len(dims)-2 and weight_nor:
                net.append(weight_norm(nn.Linear(dims[i], dims[i+1], bias, dtype=torch.float64)))
            else:
                net.append(nn.Linear(dims[i], dims[i+1], bias, dtype=torch.float64))
            net.append(nn.LeakyReLU(leaky))
        if not activate_out:
            net.pop()

        if init_zero:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

        self._net = nn.Sequential(*net)

    def forward(self, z: Tensor) -> Tensor:
        """
        Override of the torch.nn.Module method    

        Parameters
        ----------
        z
            Input data

        Returns
        -------
        Tensor
            Output data
        """
        return self._net(z)
