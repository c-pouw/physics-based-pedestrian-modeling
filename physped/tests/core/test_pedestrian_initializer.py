import logging

import numpy as np
import pandas as pd
import pytest

from physped.core.pedestrian_initializer import (
    get_initial_dynamics,
    initialize_pedestrians,
    sample_dynamics_from_trajectories,
)
from physped.utils.config_utils import initialize_hydra_config

log = logging.getLogger(__name__)


@pytest.fixture
def mock_trajs():
    trajectories = pd.DataFrame(
        {
            "xf": [0, 1, 0, 1],
            "yf": [0, 1, 0, 1],
            "uf": [0, 1, 0, 1],
            "vf": [0, 1, 0, 1],
            "xs": [0, 1, 0, 1],
            "ys": [0, 1, 0, 1],
            "us": [0, 1, 0, 1],
            "vs": [0, 1, 0, 1],
            "Pid": [0, 0, 1, 1],
            "k": [0, 1, 0, 1],
        }
    )
    return trajectories


# def test_get_initial_dynamics_from_file():
#     config = initialize_hydra_config("narrow_corridor")
#     config.params.simulation.initial_dynamics.get_from = "file"
#     filename = config.params.simulation.initial_dynamics.filename
#     config.params.simulation.initial_dynamics.filename = (
#         "outputs/narrow_corridor/" + filename
#     )
#     config.params.simulation.ntrajs = 2
#     initial_dynamics = get_initial_dynamics(config)
#     assert initial_dynamics.shape == (2, 8)


def test_get_initial_dynamics_from_point():
    config = initialize_hydra_config("narrow_corridor")
    config.params.simulation.initial_dynamics.get_from = "point"
    config.params.simulation.initial_dynamics.point = [0, 0, 0, 0, 0, 0, 0, 0]
    config.params.simulation.ntrajs = 2
    initial_dynamics = get_initial_dynamics(config)
    assert initial_dynamics.shape == (2, 8)
    assert np.all(initial_dynamics == 0)


def test_initialize_pedestrians():
    initial_dynamics = np.zeros((2, 8))
    origins = initialize_pedestrians(initial_dynamics=initial_dynamics)
    assert origins.shape == (2, 11)


def test_sample_dynamics_from_trajectories_state_0(mock_trajs):
    config = initialize_hydra_config("narrow_corridor")
    config.params.simulation.ntrajs = 2
    state_n = 0
    origins = sample_dynamics_from_trajectories(
        mock_trajs, config.params.simulation.ntrajs, state_n
    )
    assert origins.shape == (2, 8)
    assert np.all(origins == 0)


def test_sample_dynamics_from_trajectories_state_1(mock_trajs):
    config = initialize_hydra_config("narrow_corridor")
    config.params.simulation.ntrajs = 2
    state_n = 1
    origins = sample_dynamics_from_trajectories(
        mock_trajs, config.params.simulation.ntrajs, state_n
    )
    assert origins.shape == (2, 8)
    assert np.all(origins == 1)
