import numpy as np
import pytest
from omegaconf import OmegaConf

from physped.core.lattice import Lattice
from physped.utils.config_utils import initialize_hydra_config


@pytest.fixture
def mock_lattice():
    config = initialize_hydra_config("tests")
    mock_lattice = Lattice(config.params.grid.bins)
    return mock_lattice


def test_Lattice(mock_lattice):
    resolved_bins = OmegaConf.to_container(mock_lattice.bins, resolve=True)

    assert np.all(resolved_bins["x"] == np.array([-10, 0, 10]))
    assert np.all(resolved_bins["y"] == np.array([-10, 0, 10]))
    assert np.all(resolved_bins["r"] == np.array([0, 1, 2]))
    assert np.all(resolved_bins["theta"] == np.array([0, np.pi, 2 * np.pi]))
    assert np.all(resolved_bins["k"] == np.array([0, 1, 100000000000]))
    assert mock_lattice.dimensions == ("x", "y", "r", "theta", "k")

    assert np.all(mock_lattice.bin_centers["x"] == np.array([-5, 5]))
    assert np.all(mock_lattice.bin_centers["y"] == np.array([-5, 5]))
    assert np.all(mock_lattice.bin_centers["r"] == np.array([0.5, 1.5]))
    assert np.all(
        mock_lattice.bin_centers["theta"]
        == np.array([np.pi / 2, 3 * np.pi / 2])
    )
    assert np.all(
        mock_lattice.bin_centers["k"] == np.array([0.5, 50000000000.5])
    )
    assert mock_lattice.shape == (2, 2, 2, 2, 2)


def test_compute_cell_volume_shape(mock_lattice):
    """Check if the output shape matches the expected number of bins."""
    resolved_bins = OmegaConf.to_container(mock_lattice.bins, resolve=True)

    expected_shape = (
        len(resolved_bins["x"]) - 1,
        len(resolved_bins["y"]) - 1,
        len(resolved_bins["r"]) - 1,
        len(resolved_bins["theta"]) - 1,
        len(resolved_bins["k"]) - 1,
    )
    volumes = mock_lattice.compute_cell_volume()
    assert volumes.shape == expected_shape


def test_compute_cell_volume_non_negative(mock_lattice):
    """Check if the output is non-negative."""
    volumes = mock_lattice.compute_cell_volume()
    assert np.all(volumes >= 0)
