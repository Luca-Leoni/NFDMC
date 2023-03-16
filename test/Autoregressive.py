import torch

from NFDMC.Flows.Autoregressive.conditioner import MaskedConditioner
from NFDMC.Flows.Autoregressive.transformer import Affine

from hypothesis import given, settings, strategies as st

#--------------------------------

@settings(deadline=5000)
@given(var_dim=st.integers(min_value=2, max_value=100),
       trans_dim=st.integers(min_value=1, max_value=10),
       layers=st.integers(min_value=1, max_value=5))
def test_Masked_Linear(var_dim, trans_dim, layers):
    """
    Check the properties of the Masked conditioner, in particular the dimensions and the fact that h_i depends on z_{<1}(REMEMBER THE RIGHT WAY)
    """
    cond = MaskedConditioner(var_dim, trans_dim, layers, 2).to("cuda")

    z = torch.ones(3, var_dim, device="cuda")
    h = cond(z)

    assert h.shape == (3, var_dim*trans_dim)
    for i in range(var_dim-1):
        z[:, i] += 1000
        h1 = cond(z)

        assert (h1[:,:(i+1)*trans_dim] == h[:,:(i+1)*trans_dim]).all()
        h = torch.clone(h1)

@settings(deadline=5000)
@given(var_dim=st.integers(min_value=2, max_value=100),
       layers=st.integers(min_value=1, max_value=5))
def test_Affine_transformer(var_dim, layers):
    """
    Check dimensions of the net and the fact that the inverse is really an inverse
    """
    cond = MaskedConditioner(var_dim, 2, layers).to("cuda")
    trans = Affine(cond).to("cuda")

    z = torch.rand(5, var_dim, device="cuda")
    z1, log_det = trans(z)

    assert z1.shape == (5, var_dim)
    assert log_det.shape == (5,)

    z2, log_det = trans.inverse(z1)
    
    assert z2.shape == (5, var_dim)
    assert log_det.shape == (5,)
    assert torch.isclose(z2, z).all()

#--------------------------------

if __name__ == '__main__':
    test_Affine_transformer()
