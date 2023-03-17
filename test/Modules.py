import torch

from NFDMC.Modules.masked import MaskedLinear
from hypothesis import given, settings, strategies as st

@settings(deadline=5000)
@given(dim_1=st.integers(min_value=2, max_value=500),
       dim_2=st.integers(min_value=2, max_value=500),
       batch=st.integers(min_value=1, max_value=500))
def test_Masked_Linear(dim_1, dim_2, batch):
    mask = torch.tril(torch.ones(dim_2, dim_1))
    mod  = MaskedLinear(dim_1, dim_2, "lowt").to("cuda")

    assert mod.in_features == dim_1
    assert mod.out_features == dim_2
    for name, buff in mod.named_buffers():
        if name == '_mask':
            assert (buff == mask.to("cuda")).all()

    assert mod(torch.ones(batch, dim_1, device="cuda")).shape == (batch, dim_2)
