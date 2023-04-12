import torch
import torch.nn as nn

from ..Archetypes import Transformer, LTransformer
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
        res = h[:, 1::3] - h[:, ::3] - h[:, 2::3]
        res = torch.logsumexp(torch.cat((res, torch.zeros(h.shape[0], 1, device=h.device)), dim=1), dim=1)

        nan = torch.isnan(res)
        if nan.any():
            print("\nFrom PAffine transformer:")
            print(torch.arange(_.shape[0], device=nan.device)[nan])
            print(_[nan, :])
            print(h[nan, :])
            print()

        return -res



class CPAffine(LTransformer):
    """
    Defines the constrained positive Affine transformer a transformation that maps positives entries with positives values within a certain range L without the loose of flexibility of the Affine transformation
    """
    def __init__(self, UL: Tensor | float = 0.):
        r"""
        Constructor

        The transformation inside this transformer requires three parameters a, b and c so that the transformation looks like the following:
            .. math::
                \tau(z, L ,C; a, b) = c\frac{z + e^{-a}}{L + e^{-a}}, \hspace{2cm} c = \frac{C}{1 + e^b} 
        In this way one can see how if $z\in [0, L]$ than also the result is constrined to the range $[0, C]$ for every possible values of $a, b$. Also, one can see how the derivative is always between 0 and 1 so that is invertible and analitically computable.

        The values of the constrains can be selected using the set_upper_limit function of the limited transformer, and can be passed as:
            - float value, so that the same float will be used for both C and L
            - A tensor with a value for every element in the batch that will be used for both C and L
            - A tensor with two entries for every element batch defining C and L

        Parameters
        ----------
        UL
            Upper limit for the constrains
        """
        super().__init__(2, UL)


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
        L, C = self.get_constains()
        a = torch.exp(-h[:, ::2])
        c = C/(1 + torch.exp(h[:, 1::2]))

        return c * ((z + a) / (L + a)) # pyright: ignore


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
        L, C = self.get_constains()
        a = torch.exp(-h[:, ::2])
        c = C/(1 + torch.exp(h[:, 1::2]))

        return (L + a) * z / c - a

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
        L, C = self.get_constains()
        a = torch.exp(-h[:, ::2])
        c = C/(1 + torch.exp(h[:, 1::2]))

        res = torch.sum(torch.log(c / (L + a)), dim=1)

        bad = torch.isnan(res) | torch.isinf(res)
        if bad.any():
            print("\nFrom CPAffine transformer:")
            print(torch.arange(_.shape[0], device=bad.device)[bad])
            print(_[bad, :])
            print(h[bad, :])
            print()

        return res


    def get_constains(self):
        """
        Utility to retrive the constrains inside the transformation from the upeer limit inserted inside the transformer
        """
        if len(self.UL.shape) == 0:
            return self.UL, self.UL
        elif self.UL.shape[1] == 1 and len(self.UL.shape) == 2:
            return self.UL, self.UL
        elif  self.UL.shape[1] == 2 and len(self.UL.shape) == 2:
            return self.UL[:, 0:1], self.UL[:, 1:2]
        else:
            return self.UL[:, 0, ...], self.UL[:, 1, ...]



class Softplus(Transformer):
    """
    Transformer performing the softplus transformation on the input vector, the linearized and invertible version of the ReLU function
    """
    def __init__(self):
        r"""
        Constructor

        Generates a transformer that needs one parameter defining the $\beta$ of the softplus transformation, which is defined as:
            .. math::
                z' = \frac{1}{\beta}\log(1 + e^{\beta z})
        Which is easily invertible and it's derivative respect to x is simply the sigmoid function. Also, it's possible to see how such transformation maps all real numbers to positive once.
        """
        super().__init__(1)

    def forward(self, z: Tensor, h: Tensor) -> Tensor:
        """
        Override of the torch.nn.Module method

        Transform a batch of samples appling the softplus function element wise.

        Parameters
        ----------
        z
            Batch of samples
        h
            parameters of the transformation

        Returns
        -------
        Tensor
            Transformed batch
        """
        res = nn.functional.softplus(h * z) / h
        bad = (res.isnan() | res.isinf()).any(dim=1)
        if bad.any():
            print("Softplus exploded!")
            print(f"input:\n{z[bad]}")
            print(f"param:\n{h[bad]}")
        return res

    def inverse(self, z: Tensor, h: Tensor) -> Tensor:
        r"""
        Perform the inverse of the softplus element wise on the batch of samples given. Such transformation is simply given by:
            .. math::
                z' = \frac{1}{\beta}\log(e^{\beta z} - 1)

        Parameters
        ----------
        z
            Batch of samples
        h
            parameter of the transformation

        Returns
        -------
        Tensor
            Inverted batch
        """
        res = torch.where(h * z > 20., h * z, torch.log(torch.special.expm1(h * z))) / h
        bad = res.isinf().any(dim=1) | res.isnan().any(dim=1)
        if bad.any():
            print("Inverse softplus esplode!")
            print(f"input:\n{z[bad]}")
            print(f"param:\n{h[bad]}")
        return res

    def log_det(self, z: Tensor, h: Tensor) -> Tensor:
        r"""
        Computes the log determinant of the transformation, which is:
            .. math::
                \log \det J = \sum_i \log\left( \frac{1}{1 + e^{\beta_i z_i}} \right)

        Parameters
        ----------
        z
            Batch of samples
        h
            Parameter of the transformation

        Returns
        -------
        Tensor
            log determinant of the transformation for every element in the batch
        """
        res = - torch.nn.functional.softplus(- h * z).sum(dim=1)
        bad = res.isnan() | res.isinf()
        if bad.any():
            print(f"Derivata della softplus esplosa per:")
            print(z[bad])
            print(h[bad])
            print((h * z)[bad])
        return res



class Sigmoid(Transformer):
    """
    Transfomer that implements the sigmoid as a transformation applied to the entries of the samples
    """
    def __init__(self):
        r"""
        Constructor

        Construct a transformer that applies the sigmoid elementwise to the elements of the samples. The transformation will so look like:
            .. math::
                z' = \frac{1}{1 + e^{-\sigma z}}
        Where only one variable needed defining the smearing of the sigmoid.
        """
        super().__init__(1)

    def forward(self, z: Tensor, h: Tensor) -> Tensor:
        """
        Override of the torch.nn.Module method

        Apply the sigmoid function element wise to the batch of samples

        Parameters
        ----------
        z
            Batch of samples
        h
            Parameters of the transformation

        Returns
        -------
        Tensor
            Transformed samples
        """
        res = torch.sigmoid(h * z)
        bad = res.isnan().any(dim=1) | res.isinf().any(dim=1)
        if bad.any():
            print("Sigmoid exploded!")
            print(f"input:\n{z[bad]}")
            print(f"param:\n{h[bad]}")
        return res

    def inverse(self, z: Tensor, h: Tensor) -> Tensor:
        """
        Apply the inverse of the sigmoid, the logit function, to all the element of the samples in the batch.

        Parameters
        ----------
        z
            Batch of samples
        h
            Parameters of the transformation

        Returns
        -------
        Tensor
            Transformed batch of samples
        """
        return torch.logit(z) / h

    def log_det(self, z: Tensor, h: Tensor) -> Tensor:
        r"""
        Computes the log determinant of the Jacobian of the transformation, which is given by:
            .. math::
                \log \det J = \sum_i\left[ \log s(\sigma_i z_i) + \log s(-\sigma_i z_i) \right]
        where $s$ referes to the sigmoid function itself.

        Parameters
        ----------
        z
            Batch of samples
        h
            Parameters of the tramsformation

        Returns
        -------
        Tensor
            Log determinant for every sample in the batch
        """
        x = h * z
        res = -torch.sum(nn.functional.softplus(x) + nn.functional.softplus(-x), dim=1)
        bad = res.isnan() | res.isinf()
        if bad.any():
            print(f"Derivata della sigmoide esplosa per:")
            print(f"input:\n{z[bad]}")
            print(f"parameter:\n{h[bad]}")
            print(f"h * z:\n{x[bad]}")
        return res
