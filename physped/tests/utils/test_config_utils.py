import numpy as np
from omegaconf import OmegaConf

from physped.core.lattice import Lattice
from physped.utils.config_utils import initialize_hydra_config


def test_initialize_hydra_config():
    config = initialize_hydra_config("tests")
    assert config.params.env_name == "tests"


def test_resolver_cast_numpy_array():
    config = initialize_hydra_config("tests")
    config.params.grid.bins.x = "${cast_numpy_array: [-1, 1, 1]}"
    lattice = Lattice(config.params.grid.bins)
    resolved_bins = OmegaConf.to_container(lattice.bins, resolve=True)
    assert np.all(resolved_bins["x"] == np.array([-1, 1, 1]))


def test_resolver_generate_linear_bins():
    config = initialize_hydra_config("tests")
    config.params.grid.bins.x = "${generate_linear_bins: -1, 1, 1}"
    lattice = Lattice(config.params.grid.bins)
    resolved_bins = OmegaConf.to_container(lattice.bins, resolve=True)
    assert np.all(resolved_bins["x"] == np.array([-1, 0, 1]))
