import torch

from NFDMC.Flows.transformer import CPAffine

#-----------------------------

def test_CPAffine_set_constrain_float():
    trans = CPAffine()

    trans.set_upper_limit(3.)

    L, C = trans.get_constains()

    assert L == torch.tensor(3.)
    assert C == torch.tensor(3.)


def test_CPAffine_set_constrain_one_row():
    trans = CPAffine()

    trans.set_upper_limit(torch.ones(100, 1))

    L, C = trans.get_constains()

    assert (L == torch.ones(100, 1)).all()
    assert (C == torch.ones(100, 1)).all()
