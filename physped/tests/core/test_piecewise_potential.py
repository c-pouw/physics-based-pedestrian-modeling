import numpy as np

from physped.core.distribution_approximator import GaussianApproximation
from physped.core.lattice import Lattice
from physped.core.piecewise_potential import PiecewisePotential
from physped.utils.config_utils import initialize_hydra_config


def mock_piecewise_potential() -> PiecewisePotential:
    """
    Mock a piecewise potential for testing.
    """
    config = initialize_hydra_config("tests")
    lattice = Lattice(config.params.grid.bins)
    dist_approximation = GaussianApproximation()
    piecewise_potential = PiecewisePotential(lattice, dist_approximation)
    return config, piecewise_potential


def test_piecewise_potential_histograms():
    _, piecewise_potential = mock_piecewise_potential()
    assert piecewise_potential.histogram.shape == (2, 2, 2, 2, 2)
    assert isinstance(piecewise_potential.histogram, np.ndarray)
    assert np.all(isinstance(x, np.float64) for x in piecewise_potential.histogram), "Not all elements are of type np.float64"
    assert piecewise_potential.histogram_slow.shape == (2, 2, 2, 2, 2)
    assert np.all(
        isinstance(x, np.float64) for x in piecewise_potential.histogram_slow
    ), "Not all elements are of type np.float64"


def test_piecewise_potential_parametrization():
    config, piecewise_potential = mock_piecewise_potential()
    assert piecewise_potential.parametrization.shape == (2, 2, 2, 2, 2, 4, 2)
    assert np.all(
        isinstance(x, np.float64) for x in piecewise_potential.parametrization
    ), "Not all elements are of type np.float64"


def test_reparametrize_potential():
    config, piecewise_potential = mock_piecewise_potential()
    piecewise_potential.reparametrize_to_curvature(config)
    assert piecewise_potential.parametrization.shape == (2, 2, 2, 2, 2, 4, 2)
    assert np.all(
        isinstance(x, np.float64) for x in piecewise_potential.parametrization
    ), "Not all elements are of type np.float64"
