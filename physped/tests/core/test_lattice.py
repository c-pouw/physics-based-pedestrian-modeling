import numpy as np
from omegaconf import OmegaConf

from physped.core.lattice import Lattice
from physped.utils.config_utils import initialize_hydra_config


def test_Lattice():
    config = initialize_hydra_config("tests")
    lattice = Lattice(config.params.grid.bins)
    resolved_bins = OmegaConf.to_container(lattice.bins, resolve=True)

    assert np.all(resolved_bins["x"] == np.array([-10, 0, 10]))
    assert np.all(resolved_bins["y"] == np.array([-10, 0, 10]))
    assert np.all(resolved_bins["r"] == np.array([0, 1, 2]))
    assert np.all(resolved_bins["theta"] == np.array([0, np.pi, 2 * np.pi]))
    assert np.all(resolved_bins["k"] == np.array([0, 1, 100000000000]))
    assert lattice.dimensions == ("x", "y", "r", "theta", "k")

    assert np.all(lattice.bin_centers["x"] == np.array([-5, 5]))
    assert np.all(lattice.bin_centers["y"] == np.array([-5, 5]))
    assert np.all(lattice.bin_centers["r"] == np.array([0.5, 1.5]))
    assert np.all(lattice.bin_centers["theta"] == np.array([np.pi / 2, 3 * np.pi / 2]))
    assert np.all(lattice.bin_centers["k"] == np.array([0.5, 50000000000.5]))
    assert lattice.shape == (2, 2, 2, 2, 2)
