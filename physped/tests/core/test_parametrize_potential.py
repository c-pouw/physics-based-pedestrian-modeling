# import warnings

import numpy as np
import pandas as pd

from physped.core.lattice import Lattice
from physped.core.parametrize_potential import apply_periodic_angular_conditions, digitize_trajectories_to_grid
from physped.core.slow_dynamics import compute_slow_dynamics
from physped.preprocessing.trajectories import preprocess_trajectories
from physped.utils.config_utils import initialize_hydra_config

test_trajectories = {
    "xpos": [0, 1, 2, 3, 4, 0, 1, 2, 3, 0],
    "ypos": [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
    "pid": [0, 0, 0, 0, 0, 1, 1, 1, 1, 2],
    "tstep": [0, 1, 2, 3, 4, 0, 1, 2, 3, 1],
}

test_config = initialize_hydra_config("tests")


def test_apply_periodic_angular_conditions():
    df = pd.DataFrame(test_trajectories)
    df["thetaf"] = np.linspace(-np.pi, np.pi, 10)
    df["thetas"] = np.linspace(-2 * np.pi, -np.pi, 10)
    bins_min = np.pi
    bins_max = bins_min + 2 * np.pi
    lattice = Lattice(bins={"theta": np.array([bins_min, bins_max])})
    df = apply_periodic_angular_conditions(df, lattice)
    assert df.thetaf.min() >= np.pi
    assert df.thetaf.max() < 3 * np.pi
    assert df.thetas.min() >= np.pi
    assert df.thetas.max() < 3 * np.pi


def test_digitize_trajectories_to_grid():
    # ! This test needs improvements
    df = pd.DataFrame(test_trajectories)
    lattice = Lattice(test_config.params.grid.bins)
    df = preprocess_trajectories(df, config=test_config)
    df = compute_slow_dynamics(df, config=test_config)
    df = digitize_trajectories_to_grid(df, lattice)
    # print(df)
    assert "fast_grid_indices" in df.columns
    assert "slow_grid_indices" in df.columns
    # warnings.warn(UserWarning("This test needs improvements"))
