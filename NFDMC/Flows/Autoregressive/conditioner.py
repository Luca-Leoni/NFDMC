import torch
import torch.nn as nn

from ...Modules.Masked import MaskedLinear
from torch import Tensor

#---------------------------------------

class MaskedConditioner(nn.Module):
    """
    Maked conditioner where the parameters for the transformer are evaluated trhough a neural network with a series of MaskedLinear layers along with ReLU activation functions.
    """
    def __init__(self, variable_dim: int, trans_features: int, net_lenght: int = 1, hidden_multiplier: int = 1, init_zero: bool = False, bias: bool = True):
        r"""
        Constructor

        Construct the Conditioner with a series of perceptron layers with sizes defined by the variable dimensions and the number of features of the parameters needed inside the transformer. Basically it construct a function of the type:
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
        hidden_multiplier
            Scale factor that determines the dimensions of the variables in the hidden layer, in particular the hidden layers will have dimensions given by variable_dim * trans_features * hidden_multiplier. So that a net with var_dim = 2, trans_fet = 4, and hidden_mul = 2 will have the input layer of dim 2, the hidden layers of dimension 16 and the output one of dimension 8
        init_zero
            Tell if the last layer and the parameters of the first variable are initialized as zeros
        bias
            Tell if the bias is present in the masked linear layers
        """
        super().__init__()
       
        # The last variable is not used inside computations
        variable_dim -= 1
        self._trans_features = trans_features

        # Set up the dimensions and net list
        out_dim = variable_dim * trans_features
        hid_dim = variable_dim * trans_features * hidden_multiplier
        net = []

        ## Create the model

        # Adding the first layer
        net.append(MaskedLinear(variable_dim, hid_dim, mask=self._get_mask(variable_dim, hid_dim, tile_width=1, tile_lenght=trans_features*hidden_multiplier), bias=bias))

        # Temporarily increase trans_features for hidden layers
        for _ in range(net_lenght):
            # Add non linear effects
            net.append(nn.ReLU())

            # Add another Masked Layer
            net.append(MaskedLinear(hid_dim, hid_dim, mask=self._get_mask(hid_dim, hid_dim, tile_width=trans_features*hidden_multiplier, tile_lenght=trans_features*hidden_multiplier), bias=bias))

        # Add last layer
        net.append(nn.ReLU())
        net.append(MaskedLinear(hid_dim, out_dim, mask=self._get_mask(hid_dim, out_dim, tile_width=trans_features*hidden_multiplier, tile_lenght=trans_features)))

        if init_zero:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

        ## Saving model

        # Save the final result
        self.net = nn.Sequential(*net)

        # Create parameters for the features of the first variables
        self.h1  = nn.Parameter(torch.zeros(trans_features) if init_zero else torch.rand(trans_features))

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

    def _get_mask(self, in_features: int, out_features: int, tile_width: int = 1, tile_lenght: int = 1) -> Tensor:
        """
        Generates a tiled mask for the MaskedLinear layers

        Basically it generates a mask composed by tiles described as follows
            [[1, 1, 0, ..., 0],
             [1, 1, 0, ..., 0],
             [1, 1, 0, ..., 0],
             [1, 1, 1, ..., 0],
             ...
             [1, 1, 1, ..., 1]]
        Basically a kind of lower triangular matrix with a series of ones composing tiles with width given by number of rows repeated and lenght the number of ones added after a tile is complete.

        Parameters
        ----------
        in_features
            Dimensions of the input data
        out_features
            Dimesnions of the output data
        tile_width
            Width of the tiles, giving the number of rows to repeate to complite a tile
        tile_lenght
            Lenght of the tiles, giving the number of ones to add after every tile

        Returns
        -------
        Tensor
            Desired mask for the conditioner
        """
        mask = torch.ones(out_features, in_features)

        for i in range(out_features):
            mask[i * tile_lenght:(i+1) * tile_lenght, (i+1) * tile_width:] = 0

        return mask
