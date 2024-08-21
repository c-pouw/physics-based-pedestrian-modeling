import logging

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.signal import savgol_filter
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from physped.preprocessing.trajectories import transform_slow_velocity_to_polar_coordinates
from physped.utils.functions import compose_functions

log = logging.getLogger(__name__)


def integrate_slow_velocity_single_path(x0: float, us: pd.Series, dt: float) -> list:
    """Compute slow dynamics by integrating the slow velocity"""
    us = list(us)
    xs = np.zeros_like(us)
    xs[0] = x0  # Initialize xs(0) to x(0)
    for i in range(0, len(xs) - 1):
        xs[i + 1] = xs[i] + dt * us[i]
    return xs


def low_pass_filter_single_path(fast: pd.Series, tau: float, dt: float) -> list:
    """Compute slow dynamics of single trajectory with low pass filter."""
    fast = list(fast)
    slow = np.zeros_like(fast)
    slow[0] = fast[0]
    for i in range(len(fast) - 1):
        slow[i + 1] = (1 - dt / tau) * slow[i] + dt / tau * fast[i]
    return slow


SLOW_ALGORITHMS = {}


def register_slow_algorithm(name: str):
    def decorator(fn):
        SLOW_ALGORITHMS[name] = fn
        return fn

    return decorator


@register_slow_algorithm("low_pass_filter")
def low_pass_filter_all_paths(df: pd.DataFrame, **kwargs):
    grouped_paths = df.groupby("Pid")[kwargs["colname"]]
    return grouped_paths.transform(lambda x: low_pass_filter_single_path(x, kwargs["tau"], kwargs["dt"]))


@register_slow_algorithm("use_fast_dynamics")
def use_fast_dynamics(df: pd.DataFrame, **kwargs):
    return df[kwargs["colname"]]


@register_slow_algorithm("integrate_slow_velocity")
def integrate_slow_velocity(df: pd.DataFrame, **kwargs):
    slow = []
    with logging_redirect_tqdm():
        for _, traj_i in tqdm(
            df.groupby("Pid"),
            desc=f"Computing {kwargs['colname'][0]}s with {kwargs['vel_col']}",
            unit="trajs",
            total=len(df.Pid.unique()),
            miniters=100,
        ):
            x0 = traj_i[kwargs["colname"]].iloc[0]
            slow_path = integrate_slow_velocity_single_path(x0, traj_i[kwargs["vel_col"]], dt=kwargs["dt"])
            slow.extend(slow_path)
    return slow


@register_slow_algorithm("savgol_smoothing")
def savgol_smoothing(df: pd.DataFrame, **kwargs):
    slow = df.groupby("Pid")[kwargs["colname"]].transform(
        lambda x: savgol_filter(x, window_length=kwargs["window_length"], polyorder=2, deriv=0, mode="interp")
    )
    return slow


def get_slow_algorithm(name: str):
    return SLOW_ALGORITHMS.get(name)


def compute_slow_velocity(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    slow_velocity_algorithm = get_slow_algorithm(config.params.model.slow_velocities_algorithm)
    log.info("Slow velocity algorithm: %s", slow_velocity_algorithm)

    df["us"] = slow_velocity_algorithm(
        df,
        colname="uf",
        tau=config.params.model["tauu"],
        dt=config.params.model["dt"],
        window_length=config.params.minimum_trajectory_length,
    )
    df["vs"] = slow_velocity_algorithm(
        df,
        colname="vf",
        tau=config.params.model["tauu"],
        dt=config.params.model["dt"],
        window_length=config.params.minimum_trajectory_length,
    )
    return df


def compute_slow_position(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    slow_position_algorithm = get_slow_algorithm(config.params.model.slow_positions_algorithm)
    log.info("Slow position algorithm: %s", slow_position_algorithm)

    df["xs"] = slow_position_algorithm(
        df,
        colname="xf",
        vel_col="us",
        tau=config.params.model["taux"],
        dt=config.params.model["dt"],
        window_length=config.params.minimum_trajectory_length,
    )
    df["ys"] = slow_position_algorithm(
        df,
        colname="yf",
        vel_col="vs",
        tau=config.params.model["taux"],
        dt=config.params.model["dt"],
        window_length=config.params.minimum_trajectory_length,
    )
    return df


functions_to_compute_slow_dynamics = [
    compute_slow_velocity,
    transform_slow_velocity_to_polar_coordinates,
    # apply_periodic_angular_conditions,
    compute_slow_position,
]

compute_slow_dynamics = compose_functions(*functions_to_compute_slow_dynamics)


# def compute_slow_dynamics(df: pd.DataFrame, config: dict) -> pd.DataFrame:
#     dt = config.params.model["dt"]
#     tauu = config.params.model["tauu"]
#     window_length = config.params.minimum_trajectory_length
#     slow_velocity_algorithm = get_slow_algorithm(config.params.model.slow_velocities_algorithm)
#     log.info("Slow velocity algorithm: %s", slow_velocity_algorithm)

#     df["us"] = slow_velocity_algorithm(df, colname="uf", tau=tauu, dt=dt, window_length=window_length)
#     df["vs"] = slow_velocity_algorithm(df, colname="vf", tau=tauu, dt=dt, window_length=window_length)
#     df = transform_slow_velocity_to_polar_coordinates(df, config)
#     df["thetas"] = periodic_angular_conditions(df["thetas"], config.params.grid.bins["theta"])

#     taux = config.params.model["taux"]
#     slow_position_algorithm = get_slow_algorithm(config.params.model.slow_positions_algorithm)
#     log.info("Slow position algorithm: %s", slow_position_algorithm)

#     df["xs"] = slow_position_algorithm(df, colname="xf", vel_col="us", tau=taux, dt=dt, window_length=window_length)
#     df["ys"] = slow_position_algorithm(df, colname="yf", vel_col="vs", tau=taux, dt=dt, window_length=window_length)
#     return df
