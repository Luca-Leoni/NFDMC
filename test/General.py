from NFDMC.Distributions import General
from hypothesis import given, settings, strategies as st

@settings(deadline=5000)
@given(dim=st.integers(min_value = 1, max_value = 10000),
       num_sample=st.integers(min_value = 1, max_value = 10000))
def test_Gaussian_forward(dim: int, num_sample: int):
    dis = General.MultiGaussian(dim).to("cuda")

    sample, log_prob = dis(num_sample)

    assert sample.shape == (num_sample, dim)
    assert log_prob.shape == (num_sample,)
    assert (log_prob < 0).all()

@settings(deadline=5000)
@given(dim=st.integers(min_value = 1, max_value = 10000),
       num_sample=st.integers(min_value = 1, max_value = 10000))
def test_Gaussian_sample_log_prob(dim: int, num_sample: int):
    dis = General.MultiGaussian(dim).to("cuda")

    sample = dis.sample(num_sample)
    log_prob = dis.log_prob(sample)

    assert sample.shape == (num_sample, dim)
    assert log_prob.shape == (num_sample,)
    assert (log_prob < 0).all()


def test_TwoMoon_sampling():
    dist = General.TwoMoon()

    sample = dist.sample(100)

    assert sample.shape == (100, 2)
