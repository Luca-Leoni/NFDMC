import torch
import torch.nn as nn

from torch import Tensor

#----------------------------

class Conditioner(nn.Module):
    """
    Primitive class for autoregressive conditioner that can be used by external users to define it's own conditioner by passing the network they have thought to it so that it will check if has the right properties.
    """
    def __init__(self, variable_dim: int, trans_features: int, net: nn.Module | None = None):
        r"""
        Constructor

        Defines a conditioner that can be used inside the some transformer classes, in particular you can pass to this class the network you want to use as conditioner, the input variable dimension you used and the transformer features that the conditioner generates and the class will store those informations by also checking that the net posses the right properties. In particular, we assume that the function created by the network is defined as:
            .. math::
                c(\mathbb{z}) = [h_1^1, \dots, h_1^{T}, h_2^1(z_1), \dots, h_{D}^{T}(z_{<D})] = \mathbb{h}
        In this way every variable of the input vector will have trans_feature, T, number of parameters generated so that $h_i^j$ will depend only on $z_{<i}$ for every $j$.

        Parameters
        ----------
        variable_dim
            Dimension of the input variable, stored inside the class
        trans_features
            Number of parameters per dimension of the variable
        net
            Network to use as conditioner, can be also left blanck and inserted after ( mainly for implementations where the class gets inherited )
        """
        super().__init__()

        self.trans_features = trans_features
        self.variable_dim  = variable_dim

        if isinstance(net, nn.Module):
            self.set_net(net)

    def set_net(self, net: nn.Module):
        """
        Set the network used as conditioner

        Parameters
        ----------
        net
            network to use
        
        Raise
        -----
        RunTimeError
            If the network doesn't posses the right properties to be a conditioner
        """
        self._net = net
        
        self.__control_net()

    def forward(self, z: Tensor) -> Tensor:
        """
        Override of torch.nn.Module method

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

    def __control_net(self):
        """
        Performs the check on the conditioner properties

        Raises
        ------
        RuntimeError:
            If the conditioner doesn't have the right properties
        """
        self._net.eval()
        z = torch.ones(self.variable_dim)
        h = self._net(z)
        for i in range(self.variable_dim-1):
            z[:,i] += 1000
            h1 = self._net(z)

            if not (h1[:, :(i+1) * self.trans_features] == h[:, :(i+1) * self.trans_features]).all():
                raise RuntimeError("The net inserted in the conditioner does not satisfy the needed properties!")


class Transformer(nn.Module):
    """
    Archetype class that defines a transformer inside the module
    """
    def __init__(self, trans_features: int): 
        r"""
        Constructor

        Create a transformer function $\tau$ that has a general form of the following type:
            .. math::
                z'_i = \tau(z_i; \mathbb{h}(z_{<i}))
        basically a transformation applied to every member of the input variable depending also on a vector $\mathbb{h}$ that is computed through the conditioner and depends on previous variables so that the jacobian is lower triangular.

        Parameters
        ----------
        trans_features
            Number of parameters created per dimension inside input variable
        """
        super().__init__()

        self.trans_features = trans_features

    def forward(self, z: Tensor, *h: Tensor) -> Tensor: # pyright: ignore
        """
        Override of the torch.nn.Module method

        Takes as input both the input data and the parameters and gives out the transformed variable

        Parameters
        ----------
        z
            Input variable
        h
            Parameters on that variable

        Returns
        -------
        Tensor
            Output variable

        Raises
        ------
        NotImplementedError:
            Needs to be implemented by the user
        """
        raise NotImplementedError()

    def inverse(self, z: Tensor, *h: Tensor) -> Tensor: # pyright: ignore
        r"""
        Definition of the inverse $\tau^{-1}$

        Takes a transformed variable and the parameter of the transformation and performes the inverse

        Parameters
        ----------
        z
            Transformed variable
        h
            Parameters on the untrasformed variable

        Returns
        -------
        Tensor
            Untransformed variable

        Raises
        ------
        NotImplementedError:
            Needs to be implemented by the user
        """
        raise NotImplementedError()

    def log_det(self, z: Tensor, *h: Tensor) -> Tensor: # pyright: ignore
        """
        Computes the log determinante of the forward transformation

        Parameters
        ----------
        z
            Variable where to compute the jacobian
        h
            Parameters that defines the transformation

        Returns
        -------
        Tensor
            log determinant of the jacobian

        Raises
        ------
        NotImplementedError:
            Needs to be implemented by the user
        """
        raise NotImplementedError()
