import torch
import torch.nn as nn

from ...Modules.Masked import MaskedLinear
from torch import Tensor

class MaskedConditioner(nn.Module):
    """
    Maked conditioner where the parameters for the transformer are evaluated trhough a neural network with a series of MaskedLinear layers along with ReLU activation functions.
    """
    def __init__(self, features: list[int], trans_features: int, bias: bool = True):
        """
        Constructor

        Construct the Conditioner with a series of perceptron layers with sizes given inside a list, also the number of features of the parameters inside transformers are needed.

        Parameters
        ----------
        features
            List of features of the layers inside the neural network
        trans_features
            Number of features inside the parameter in the transformer function
        bias
            Tell if the bias is present
        """
        super().__init__()

        net = []
        for i in range(len(features) - 1):
            # Define the mask with wanted dimension and create the MaskedLinear layer from it
            mask = torch.tril(torch.ones(features[i+1], features[i]), diagonal=-1)
            net.append(MaskedLinear(mask, bias))

            # Add a non linear function for expressivity
            net.append(nn.ReLU())
        # Eliminate last ReLU on the output layer 
        net.pop()

        # Save the final result
        self.net = nn.Sequential(*net)

    def forward(self, x: Tensor) -> Tensor:
        """
        Override of the torch.nn.Module method

        Parameters
        ----------
        x
            Input data

        Returns
        -------
        Tensor
            Output data
        """
        return self.net(x)
