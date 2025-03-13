import numpy as np
import pytest

from physped.core.distribution_approximator import GaussianApproximation
from physped.core.lattice import Lattice
from physped.core.piecewise_potential import PiecewisePotential
from physped.utils.config_utils import initialize_hydra_config


@pytest.fixture
def mock_config():
    config = initialize_hydra_config("tests")
    return config


@pytest.fixture
def mock_piecewise_potential(mock_config) -> PiecewisePotential:
    """
    Mock a piecewise potential for testing.
    """
    lattice = Lattice(mock_config.params.grid.bins)
    dist_approximation = GaussianApproximation()
    piecewise_potential = PiecewisePotential(lattice, dist_approximation)
    return piecewise_potential


def test_piecewise_potential_histograms(mock_piecewise_potential):
    assert mock_piecewise_potential.histogram.shape == (2, 2, 2, 2, 2)
    assert isinstance(mock_piecewise_potential.histogram, np.ndarray)
    assert np.all(
        isinstance(x, np.float64) for x in mock_piecewise_potential.histogram
    ), "Not all elements are of type np.float64"
    assert mock_piecewise_potential.histogram_slow.shape == (2, 2, 2, 2, 2)
    assert np.all(
        isinstance(x, np.float64)
        for x in mock_piecewise_potential.histogram_slow
    ), "Not all elements are of type np.float64"


def test_piecewise_potential_parametrization(mock_piecewise_potential):
    assert mock_piecewise_potential.parametrization.shape == (
        2,
        2,
        2,
        2,
        2,
        4,
        2,
    )
    assert np.all(
        isinstance(x, np.float64)
        for x in mock_piecewise_potential.parametrization
    ), "Not all elements are of type np.float64"


def test_reparametrize_potential(mock_config, mock_piecewise_potential):
    mock_piecewise_potential.reparametrize_to_curvature(mock_config)
    assert mock_piecewise_potential.parametrization.shape == (
        2,
        2,
        2,
        2,
        2,
        4,
        2,
    )
    assert np.all(
        isinstance(x, np.float64)
        for x in mock_piecewise_potential.parametrization
    ), "Not all elements are of type np.float64"
