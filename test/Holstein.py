import torch

from NFDMC.Distributions import Holstein
from hypothesis import given, settings, strategies as st

@settings(deadline=5000)
@given(tm_fly = st.floats(min_value = 1E-6, max_value = 1E6, allow_infinity = False, allow_nan = False),
       max_order   = st.integers(min_value = 0, max_value = 10000).filter(lambda x: x % 2 == 0),
       num_samples = st.integers(min_value = 1, max_value = 10000))
def test_holstein_base_generation(tm_fly: float, max_order: int, num_samples: int):
    """
    Test the generation of diagrams from base distribution

    the things that we are testing are the following:
        - The shape of the samples genereted is effectivelly equal to (num_sample, max_order + 1)
        - The number of log probabilities computes is equal to the number of samples
        - The anihlations times are larger than the creation ones
        - The order at the top is effectivelly an integer by seing how is equal to result of floor function
        - The orders are all less or equal to the maximum one
    """
    dis = Holstein.Base(tm_fly, max_order).to("cuda")

    dia, log_p = dis(num_samples)

    assert dia.shape == (num_samples, max_order+1)
    assert log_p.shape == (num_samples,)
    assert (dia[:,1::2] <= dia[:,2::2]).all()
    assert (torch.floor(dia[:, 0]) == dia[:,0]).all()
    assert (dia[:,0] <= max_order).all()
