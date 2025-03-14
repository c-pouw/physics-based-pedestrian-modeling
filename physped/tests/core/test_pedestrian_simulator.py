from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from physped.core.pedestrian_simulator import simulate_pedestrians
from physped.io.readers import read_piecewise_potential_from_file
from physped.utils.config_utils import initialize_hydra_config


@pytest.fixture
def mock_config():
    config = initialize_hydra_config("narrow_corridor")
    return config


@pytest.fixture
def mock_piecewise_potential_from_file():
    test_piecewise_potential_path = Path.cwd() / (
        "physped/tests/data/"
        "piecewise_potential_with_dxy0.2_r-0-5-10-15-20-25-30_ntheta8.pickle"
    )
    piecewise_potential = read_piecewise_potential_from_file(
        test_piecewise_potential_path
    )
    return piecewise_potential


# @pytest.fixture
# def mock_trajectories():
#     trajectories = pd.DataFrame(
#         {
#             "xf": [0, 1, 2, 3, 4],
#             "yf": [0, 1, 2, 3, 4],
#             "uf": [0, 0, 0, 0, 0],
#             "vf": [0, 0, 0, 0, 0],
#             "xs": [0, 1, 2, 3, 4],
#             "ys": [0, 1, 2, 3, 4],
#             "us": [0, 0, 0, 0, 0],
#             "vs": [0, 0, 0, 0, 0],
#             "Pid": [0, 0, 0, 0, 0],
#             "k": [0, 0, 0, 0, 0],
#         }
#     )
#     return trajectories


def test_simulate_trajectories(
    mock_config, mock_piecewise_potential_from_file
):
    mock_config.params.simulation.initial_dynamics.get_from = "point"
    mock_config.params.simulation.ntrajs = 1
    simulated_trajectories = simulate_pedestrians(
        mock_piecewise_potential_from_file, mock_config
    )
    assert simulated_trajectories is not None
    assert isinstance(simulated_trajectories, pd.DataFrame)
    expected_columns = set(
        [
            "xf",
            "yf",
            "uf",
            "vf",
            "xs",
            "ys",
            "us",
            "vs",
            "t",
            "k",
            "Pid",
            "piece_id",
            "rf",
            "thetaf",
            "rs",
            "thetas",
        ]
    )
    assert expected_columns.issubset(set(simulated_trajectories.columns))
    assert np.all(simulated_trajectories.iloc[0] == 0)
    assert np.all(simulated_trajectories.Pid == 0)
