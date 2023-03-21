import torch

from NFDMC.Distributions import diagrams
from hypothesis import given, settings, strategies as st

@given(st.integers(min_value=2, max_value=100).filter(lambda x: not x % 2))
def test_Holstein(max_order: int):
    dis = diagrams.Holstein(1, 0.2, 0.5)

    z = torch.rand(2, max_order+1)
    z[:, 0] = torch.randint(low=0, high=max_order, size=(2,))

    # Construct a diagram
    for i in range(1, max_order, 2):
        z[0, i+1] += z[0, i]

    # Set second as non diagram
    z[1, 2] = 0

    log_p = dis.log_prob(z)

    assert log_p.shape == (2,)
    assert log_p[1] == -1000000
    assert log_p[0] != -1000000
