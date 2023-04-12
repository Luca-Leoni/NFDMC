import torch

from NFDMC.Flows import Autoregressive, Coupling, NVPCoupling
from NFDMC.Flows.conditioner import MaskedConditioner
from NFDMC.Flows.transformer import Affine, Softplus
from NFDMC.Modules.nets import MLP

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
    flow = Autoregressive(Affine(), MaskedConditioner(var_dim, 2, layers)).to("cuda")

    z = torch.rand(5, var_dim, device="cuda")
    z1, log_det = flow(z)

    assert z1.shape == (5, var_dim)
    assert log_det.shape == (5,)

    z2, log_det = flow.inverse(z1)
    
    assert z2.shape == (5, var_dim)
    assert log_det.shape == (5,)
    assert torch.isclose(z2, z).all()


@settings(deadline=5000)
@given(var_dim=st.integers(min_value=2, max_value=100),
       num_sample=st.integers(min_value=1, max_value=1000))
def test_Affine_Coupling(var_dim, num_sample):
    """
    Check dimensions of the net and the fact that the inverse is really an inverse
    """
    split = [var_dim // 2, var_dim - (var_dim // 2)]
    flow = Coupling(Affine(), MLP(split[0], 30, 30, 2 * split[1]), split=split).to("cuda")

    z = torch.rand(num_sample, var_dim, device="cuda", dtype=torch.float64)
    z1, log_det = flow(z)
    z2, log_det_inv = flow.inverse(z1)

    assert z1.shape == (num_sample, var_dim)
    assert log_det.shape == (num_sample,)
    assert z2.shape == (num_sample, var_dim)
    assert (log_det == -log_det_inv).all()
    assert torch.isclose(z2, z, rtol=0., atol=1e-5).all()


@settings(deadline=5000)
@given(var_dim=st.integers(min_value=2, max_value=100),
       num_sample=st.integers(min_value=1, max_value=1000))
def test_Softplus_NVPCoupling(var_dim, num_sample):
    """
    Check dimensions of the net and the fact that the inverse is really an inverse
    """
    flow = NVPCoupling(Softplus(), split=var_dim).to("cuda")

    z = torch.rand(num_sample, var_dim, device="cuda", dtype=torch.float64)
    z1, log_det = flow(z)
    z2, log_det_inv = flow.inverse(z1)

    assert z1.shape == (num_sample, var_dim)
    assert log_det.shape == (num_sample,)
    assert z2.shape == (num_sample, var_dim)
    assert torch.isclose(log_det, -log_det_inv, rtol=0, atol=1e-5).all()
    assert torch.isclose(z2, z, rtol=0., atol=1e-5).all()

#--------------------------------

if __name__ == '__main__':
    test_Affine_transformer()
