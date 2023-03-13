import torch

from NFDMC.Flows.Autoregressive.conditioner import MaskedConditioner


def test_Masked_Linear():
    cond = MaskedConditioner([10, 10, 10]).to("cuda")

    assert cond(torch.ones(10, device="cuda")).shape == (10,)
