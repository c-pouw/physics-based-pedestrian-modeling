import logging
from functools import reduce
from typing import Any, Callable

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.signal import savgol_filter

from physped.utils.functions import cartesian_to_polar_coordinates

log = logging.getLogger(__name__)


def rename_columns(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """
    Rename columns of a DataFrame.

    Args:
    - df: The DataFrame to rename the columns of.
    - colnames: A dictionary with the old column names as keys and the new column names as values.

    Returns:
    - The DataFrame with the columns renamed.
    """
    colnames = config.params.get("colnames", {})
    inverted_colnames = {v: k for k, v in colnames.items()}
    df.rename(columns=inverted_colnames, inplace=True)
    log.info("Columns renamed to %s", list(df.columns))
    return df


def prune_short_trajectories(trajectories: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """Remove short trajectories that are most likely noise.

    The threshold for the minimum trajectory length is set in the configuration object
    under the key config.params.minimum_trajectory_length.

    Args:
        trajectories: The DataFrame containing the trajectories.
        config: The configuration object.

    Returns:
        The DataFrame without short trajectories.
    """
    trajectories["traj_len"] = trajectories.groupby(["Pid"])["Pid"].transform("size")
    trajectories = trajectories[trajectories.traj_len > config.params.minimum_trajectory_length].copy()
    log.info("Short trajectories with less than %s observations removed.", config.params.minimum_trajectory_length)
    return trajectories


def add_trajectory_index(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """Add a column with the observation/step numbers of each trajectory.

    Args:
    - df: The DataFrame to add the trajectory step/observation to.
    - config: The configuration object.

    Returns:
    - The DataFrame with the trajectory step/observation added.
    """
    # pid_col, time_col = params.colnames.Pid, params.colnames.time
    pid_col, time_col = "Pid", "time"
    df.sort_values(by=[pid_col, time_col], inplace=True)
    df["k"] = df.groupby(pid_col)[pid_col].transform(lambda x: np.arange(x.size))
    log.info("Trajectory step added.")
    return df


def compute_velocity_from_positions(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """Add velocity to dataframe with trajectories.

    This function calculates the velocity of each pedestrian in the input DataFrame
    based on their position data. The velocity is calculated using the Savitzky-Golay
    filter with a window size of 49 and a polynomial order of 1.

    Args:
        df: The input DataFrame with the trajectories

    Returns:
        The trajectories DataFrame with velocity columns added.
    """
    uf = config.params.colnames.get("uf", None)
    vf = config.params.colnames.get("vf", None)

    if uf is not None or vf is not None:
        log.warning("Columns for velocity found in parameters, no need to calculate velocity.")
        return df

    groupby = "Pid"
    xpos = "xf"
    ypos = "yf"
    pos_to_vel = {"xf": "uf", "yf": "vf"}
    window_length = config.params.velocity_window_length
    # window_length = parameters.minimum_trajectory_length - 1
    for direction in [xpos, ypos]:
        df.loc[:, pos_to_vel[direction]] = df.groupby([groupby])[direction].transform(
            lambda x: savgol_filter(x, window_length=window_length, polyorder=1, deriv=1, mode="interp") * config.params.fps
        )
    log.info("Velocities 'uf' and 'vf' added.")
    return df


def transform_velocity_to_polar_coordinates(df: pd.DataFrame, config: DictConfig, dynamics: str = "f") -> pd.DataFrame:
    """Add columns with the velocity in polar coordinates to the DataFrame.

    Requires the columns 'u' and 'v' for the velocity in x and y direction, respectively.

    Args:
    - df: The DataFrame to add polar coordinates to.
    - dynamics: The dynamics to add polar coordinates for 'f' for fast or 's' for slow dynamics.
        defauls to fast dynamics.

    Returns:
    - The DataFrame with additional columns for the velocity in polar coordinates.
    """
    ucol = "u" + dynamics
    vcol = "v" + dynamics
    rcol = "r" + dynamics
    thetacol = "theta" + dynamics
    df[rcol], df[thetacol] = cartesian_to_polar_coordinates(df[ucol], df[vcol])
    log.info("Velocity in polar coordinates 'r' and 'theta' added for dynamics %s.", dynamics)
    return df


# def add_acceleration(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
#     framerate = parameters.fps
#     groupby = "Pid"
#     xcol = "uf"
#     ycol = "vf"
#     new_col = {"uf": "axf", "vf": "ayf"}
#     window_length = parameters.velocity_window_length
#     # window_length = parameters.minimum_trajectory_length - 1
#     for direction in [xcol, ycol]:
#         df.loc[:, new_col[direction]] = df.groupby([groupby])[direction].transform(
#             lambda x: savgol_filter(x, window_length=window_length, polyorder=2, deriv=2, mode="interp") * framerate
#         )
#     return df

Composable = Callable[[Any], Any]


def compose(*functions: Composable) -> Composable:
    return lambda x, **kwargs: reduce(lambda df, fn: fn(df, **kwargs), functions, x)


preprocessing_functions = [
    rename_columns,
    prune_short_trajectories,
    add_trajectory_index,
    compute_velocity_from_positions,
    transform_velocity_to_polar_coordinates,
]

preprocess_trajectories = compose(*preprocessing_functions)

# def preprocess_trajectories(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
#     filepath = Path.cwd().parent / config.filename.preprocessed_trajectories

#     # TODO : Move to separate function
#     if config.read.preprocessed_trajectories:
#         log.debug("Configuration 'read.preprocessed_trajectories' is set to True.")
#         try:
#             preprocessed_trajectories = read_trajectories_from_path(filepath)
#             log.warning("Preprocessed trajectories read from file.")
#             # log.debug("Filepath %s", filepath.relative_to(config.root_dir))
#             return preprocessed_trajectories
#         except FileNotFoundError as e:
#             log.error("Preprocessed trajectories not found: %s", e)

#     log.info("Start preprocessing of the recorded trajectories.")
#     # TODO : Use columnnames from parameters instead of renaming
#     df = rename_columns(df, config)

#     df = prune_short_trajectories(df, config)

#     df = add_trajectory_index(df, config)

#     df = add_velocity(df, config)

#     # axf = parameters.colnames.get("axf", None)
#     # ayf = parameters.colnames.get("ayf", None)
#     # if axf is None or ayf is None:
#     #     log.warning("Columns for acceleration not found in parameters. Calculating acceleration.")
#     #     df = add_acceleration(df, parameters)
#     #     log.info("Acceleration added.")

#     df = transform_velocity_to_polar_coordinates(df, dynamics="f")

#     # if parameters.intermediate_save.preprocessed_trajectories:
#     if config.save.preprocessed_trajectories:
#         log.debug("Configuration 'save.preprocessed_trajectories' is set to True.")
#         save_trajectories(df, Path.cwd().parent, config.filename.preprocessed_trajectories)
#     return df


# def filter_trajectories_by_velocity(df: pd.DataFrame) -> pd.DataFrame:
#     """Filter trajectories by their average velocity."""
#     log.info("Start filtering trajectories by velocity.")
#     umin = 0.2
#     df = df[df.groupby("Pid")["uf"].transform("mean") > umin]
#     log.info("Trajectories filtered by velocity.")
#     return df
