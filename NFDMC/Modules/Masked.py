import torch.nn as nn

from torch import Tensor

class MaskedLinear(nn.Linear):
    """
    Dense linear layer masked in order to eliminate varius connection inside the layer making it sparse
    """
    def __init__(self, mask: Tensor, bias: bool = True):
        """
        Constructor

        Parameters
        ----------
        mask
            Mask to apply to the dense layer, from it's dimesions the layers'sizes are deduced
        bias
            Bias inside the linear layer
        """
        super().__init__(mask.shape[0], mask.shape[1], bias)
        
        self.register_buffer("_mask", mask)


    def forward(self, x: Tensor) -> Tensor:
        """
        Override of the forward method of torch.nn.Module

        Perfor the needed matrix multiplication from input to output as usual in the linear layer but before doing it the weights are multiplied by the mask wanted

        Parameters
        ----------
        x
            Input data

        Returns
        -------
        Tensor
            Output data
        """
        return nn.functional.linear(x, self._mask.transpose(0, 1) * self.weight, self.bias) # pyright: ignore
