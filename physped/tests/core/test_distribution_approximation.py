from scipy.stats import norm

from physped.core.distribution_approximator import GaussianApproximation


def test_gaussian_approximation():
    dist_approximation = GaussianApproximation()
    assert dist_approximation.fit_dimensions == ("x", "y", "u", "v")
    assert dist_approximation.fit_parameters == ("mu", "sigma")
    assert dist_approximation.function == norm.fit
