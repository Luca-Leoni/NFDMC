import torch

from ..Archetypes import Transformer, Diagrammatic
from torch import Tensor

#-------------------------------

class Affine(Transformer):
    """
    Definition of the affine transformer as a basic linear transformation
    """
    def __init__(self):
        """
        Constructor

        Defines the affine coupling transformation as:
            .. math::
                z'_i = exp(h_0^i) * z_i + h_1^i
        So that both log det and inverse are analitically known, also the number of parameters needed for variable dimension is 2.
        """
        super().__init__(2)

    def forward(self, z: Tensor, h: Tensor) -> Tensor:
        """
        Override of the torch.nn.Module method

        Parameters
        ----------
        z
            Input variable
        h
            Parameters of the transformation

        Returns
        -------
        Tensor
            Output variable
        """
        return torch.exp(h[:, ::2]) * z + h[:, 1::2]

    def inverse(self, z: Tensor, h: Tensor) -> Tensor:
        """
        Inverse transformation evaluated through:
            .. math::
                z_i = (z_i' - h_1^i) * exp(-h_0^i)     

        Parameters
        ----------
        z
            Transformed variable
        h
            Parameters that defines the transformation

        Returns
        -------
        Tensor
            Untransformed variable
        """
        return (z - h[:, 1::2]) * torch.exp(-h[:, ::2])

    def log_det(self, _: Tensor, h: Tensor) -> Tensor:
        r"""
        Compute the log determinant of the Jacobian which ends up in being simply:
            .. math::
                \log\det J = \sum_i h_0^i

        Parameters
        ----------
        _
            Input variable, not use in this case
        h
            Parameter list evaluated using the input variable

        Returns
        -------
        Tensor
            Log determinant
        """
        return torch.sum(h[:, ::2], dim = 1)


class PAffine(Transformer):
    """
    Defines the positive Affine transformer a transformation that maps positives entries with positives values without the loose of flexibility of the Affine transformation
    """
    def __init__(self):
        r"""
        Constructor

        The transformation inside this transformer requires three parameters a, b and c so that the transformation looks like the following:
            .. math::
                \tau(z; a, b, c) = \frac{e^a z + e^b}{e^a + e^{b-c}}
        In this way one can see how if $z>0$ than also the result is positive and one can see how also the derivative is always between 0 and 1 so that is invertible and analitically computable.
        """
        super().__init__(3)


    def forward(self, z: Tensor, h: Tensor) -> Tensor:
        """
        Override of the torch.nn.Module method

        Parameters
        ----------
        z
            Input variable
        h
            Parameters of the transformation

        Returns
        -------
        Tensor
            Output variable
        """
        a = torch.exp(h[:, ::3])
        b = torch.exp(h[:, 1::3])
        c = torch.exp(h[:, 2::3])

        return (a * z + b) / (a + b/c)

    def inverse(self, z: Tensor, h: Tensor) -> Tensor:
        r"""
        Inverse transformation evaluated through:
            .. math::
                \tau^{-1}(z; a, b, c) = (1 + e^{b-c-a})z - e^{b-a}

        Parameters
        ----------
        z
            Transformed variable
        h
            Parameters that defines the transformation

        Returns
        -------
        Tensor
            Untransformed variable
        """
        a = torch.exp(h[:, ::3])
        b = torch.exp(h[:, 1::3])
        c = torch.exp(h[:, 2::3])

        return ((a + b/c)*z - b) / a

    def log_det(self, _: Tensor, h: Tensor) -> Tensor:
        r"""
        Compute the log determinant of the Jacobian which ends up in being simply:
            .. math::
                \log\det J = -\sum_i \log(1 + e^{b_i-c_i-a_i})

        Parameters
        ----------
        _
            Input variable, not use in this case
        h
            Parameter list evaluated using the input variable

        Returns
        -------
        Tensor
            Log determinant
        """
        a = torch.exp(h[:, ::3])
        b = torch.exp(h[:, 1::3])
        c = torch.exp(h[:, 2::3])

        return -torch.sum(torch.log(1 + b/(a*c)), dim=1)



class CPAffine(Transformer):
    """
    Defines the constrained positive Affine transformer a transformation that maps positives entries with positives values within a certain range L without the loose of flexibility of the Affine transformation
    """
    def __init__(self):
        r"""
        Constructor

        The transformation inside this transformer requires three parameters a, b and c so that the transformation looks like the following:
            .. math::
                \tau(z; a, b, c, L) = \frac{e^a z + e^b}{e^a + e^{b}/d}, \hspace{2cm} d = \frac{L}{1 + e^c} 
        In this way one can see how if $z>0$ than also the result is constrined to the range $[0, L]$ for every possible values of $a, b$ and $c$. Also, one can see how the derivative is always between 0 and 1 so that is invertible and analitically computable.
        """
        super().__init__(4)


    def forward(self, z: Tensor, h: Tensor) -> Tensor:
        """
        Override of the torch.nn.Module method

        Parameters
        ----------
        z
            Input variable
        h
            Parameters of the transformation

        Returns
        -------
        Tensor
            Output variable
        """
        a = torch.exp(h[:, ::4])
        b = torch.exp(h[:, 1::4])
        c = h[:, 3::4]/(1 + torch.exp(h[:, 2::4]))

        return (a * z + b) / (a + b/c)


    def inverse(self, z: Tensor, h: Tensor) -> Tensor:
        r"""
        Inverse transformation evaluated through:
            .. math::
                \tau^{-1}(z; a, b, c, L) = (1 + e^{b-a}/d)z - e^{b-a}, \hspace{2cm} d = \frac{L}{1 + e^c} 

        Parameters
        ----------
        z
            Input variable
        h
            Parameters that defines the transformation

        Returns
        -------
        Tensor
            Untransformed variable
        """
        a = torch.exp(h[:, ::4])
        b = torch.exp(h[:, 1::4])
        c = h[:, 3::4]/(1 + torch.exp(h[:, 2::4]))

        return ((a + b/c)*z - b) / a

    def log_det(self, _: Tensor, h: Tensor) -> Tensor:
        r"""
        Compute the log determinant of the Jacobian which ends up in being simply:
            .. math::
                \log\det J = -\sum_i \log(1 + e^{b_i-a_i}/d_i), \hspace{2cm} d_i = \frac{L_i}{1 + e^{c_i}}

        Parameters
        ----------
        _
            Input variable, not used here
        h
            Parameter list evaluated using the input variable

        Returns
        -------
        Tensor
            Log determinant
        """
        a = torch.exp(h[:, ::4])
        b = torch.exp(h[:, 1::4])
        c = h[:, 3::4]/(1 + torch.exp(h[:, 2::4]))

        return -torch.sum(torch.log(1 + b/(a*c)), dim=1)
