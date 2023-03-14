import torch
import torch.nn as nn

from ...Modules.Masked import MaskedLinear
from torch import Tensor

class MaskedConditioner(nn.Module):
    """
    Maked conditioner where the parameters for the transformer are evaluated trhough a neural network with a series of MaskedLinear layers along with ReLU activation functions.
    """
    def __init__(self, variable_dim: int, trans_features: int, net_lenght: int = 1, bias: bool = True):
        r"""
        Constructor

        Construct the Conditioner with a series of perceptron layers with sizes given inside a list, also the number of features of the parameters inside transformers are needed. Basically it assumes to construct a function of the type:
            .. math::
                c(\mathbb{z}) = [h_1^1, \dots, h_1^{T}, h_2^1(z_1), \dots, h_{D}^{T}(z_{<D})] = \mathbb{h}
        In this way every variable of the input vector will have trans_feature, T, number of parameters generated so that $h_i^j$ will depend only on $z_{<i}$ for every $j$.

        Parameters
        ----------
        variable_dim
            Dimension of the random variable vector in input to the conditioner
        trans_features
            Number of parameter needed for one variable in the transformer function automatically defines the output layer dimensions as variable_dim * trans_features
        net_lenght
            Tells how many layers should the net posses, having that every added layer is composed by a ReLU followed by a MaskedLinear of dimensions (out_dim, out_dim) with out_dim = variable_dim * trans_features
        bias
            Tell if the bias is present
        """
        super().__init__()
       
        # The last variable is not used inside computations
        variable_dim -= 1
        self._trans_features = trans_features

        # Set up the dimensions and net list
        out_dim = variable_dim * trans_features
        net = []

        # Adding the first layer
        net.append(MaskedLinear(variable_dim, out_dim, mask=self._get_mask(variable_dim, out_dim, inp_layer=False), bias=bias))
        for _ in range(net_lenght):
            # Add non linear effects
            net.append(nn.ReLU())

            # Add another Masked Layer
            net.append(MaskedLinear(out_dim, out_dim, mask=self._get_mask(out_dim, out_dim, inp_layer=True), bias=bias))

        # Save the final result
        self.net = nn.Sequential(*net)

        # Create parameters for the features of the first variables
        self.h1  = nn.Parameter(torch.rand(trans_features))

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
        h = self.net(x[:,:-1])
        return torch.cat( (self.h1.expand(x.shape[0], self._trans_features), h), dim=1)

    def _get_mask(self, in_features: int, out_features: int, inp_layer: bool) -> Tensor:
        """
        Generates the wanted mask for the MaskedLinear layers inside the architecture

        In particular the mask we want is composed by a connection that allows the parameters $h_i^j$ to depend only on $z_{<i}$ so that a mask of the following type is needed
            [[1, 0, 0, ..., 0],
             [1, 0, 0, ..., 0],
             [1, 1, 0, ..., 0],
             ...
             [1, 1, 1, ..., 1]]
        Basically a kind of upper triangular matrix with every column repited a number of times equal to the number of transformer parameters, in the case in figure the parameters were 2. In the case the mask needs to be created for the input layer, where the matrix is not squared but the input features are less the rows do not need to be repeated.

        Parameters
        ----------
        in_features
            Dimensions of the input data
        out_features
            Dimesnions of the output data
        inp_layer
            Tells if the mask is for the input layer because in that case the repeating of the same column is not needed

        Returns
        -------
        Tensor
            Desired mask for the conditioner
        """
        mask = torch.ones(out_features, in_features)
        rep  = self._trans_features if inp_layer else 1

        for i in range(out_features):
            mask[i*self._trans_features:(i+1)*self._trans_features, (i+1)*rep:] = 0

        return mask
