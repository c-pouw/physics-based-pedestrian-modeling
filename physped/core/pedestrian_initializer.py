import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def get_initial_dynamics(config: dict) -> np.ndarray:
    ntrajs = config.params.simulation.ntrajs
    match config.params.simulation.initial_dynamics.get_from:
        case "file":
            folderpath = Path.cwd()  # / "initial_dynamics"
            filename = (
                folderpath / config.params.simulation.initial_dynamics.filename
            )
            initial_dynamics = np.load(filename)
            if initial_dynamics.shape[0] < ntrajs:
                raise ValueError(
                    f"Number of trajectories in "
                    f"{config.params.simulation.initial_dynamics.filename} is"
                    f"less than {ntrajs}"
                )
            initial_dynamics = initial_dynamics[:ntrajs, :]
            return initial_dynamics
        case "point":
            point = np.array(config.params.simulation.initial_dynamics.point)
            initial_dynamics = np.tile(point, (ntrajs, 1))
            return initial_dynamics
        case _:
            raise ValueError(
                f"Unknown initial_dynamics.get_from: "
                f"{config.params.simulation.initial_dynamics.get_from}"
            )


def initialize_pedestrians(initial_dynamics: np.ndarray) -> np.ndarray:
    """Add time, step, and Pid to the initial dynamics

    Args:
        initial_dynamics: The initial dynamics of the pedestrians expressed as
        [xf, yf, uf, vf, xs, ys, us, vs]

    Returns:
        The initial dynamics with the time, step, and Pid added
        [xf, yf, uf, vf, xs, ys, us, vs, t, k, Pid]
    """
    pedestrian_initializations = np.hstack(
        (
            initial_dynamics,
            np.zeros(
                (initial_dynamics.shape[0], 2)
            ),  # Add start_time and start_k
            np.arange(initial_dynamics.shape[0])[:, None],  # Add Pids
        )
    )
    return pedestrian_initializations


def sample_dynamics_from_trajectories(
    trajectories: pd.DataFrame,
    n_trajs: int,
    state_n: int,
) -> np.ndarray:
    if state_n < -1:
        raise ValueError("State n must be >= -1")
    elif state_n == -1:
        # Sample from random point along each path
        states_to_sample_from = (
            trajectories.groupby("Pid")
            .apply(lambda x: x.sample(1))
            .reset_index(drop=True)
        )
    else:  # state_n >= 0:
        # Sample from state n along each path
        states_to_sample_from = trajectories[
            trajectories["k"] == state_n
        ].copy()

    sampled_states = states_to_sample_from[
        ["xf", "yf", "uf", "vf", "xs", "ys", "us", "vs"]
    ].sample(n=n_trajs, replace=True)
    log.info("Sampled %d origins from the input trajectories.", n_trajs)
    return sampled_states.to_numpy()


# def potential_defined_at_slow_state(
#     paths: pd.DataFrame, piecewise_potential: PiecewisePotential
# ) -> pd.DataFrame:
#     # REQUIRED: Dataframe must have a column with the slow grid indices
#     # TODO : Move this function to another module.
#     # TODO : Change the slow_grid_indices to lists rather than tuples
#     indices = np.array(list(paths["slow_grid_indices"]))
#     potential_defined = np.where(
#         np.isnan(piecewise_potential.parametrization), False, True
#     )
#     # All free parameters must be defined
#     potential_defined = np.all(potential_defined, axis=(-2, -1))
#     paths["potential_defined"] = potential_defined[
#         indices[:, 0],
#         indices[:, 1],
#         indices[:, 2],
#         indices[:, 3],
#         indices[:, 4],
#     ]
#     return paths
