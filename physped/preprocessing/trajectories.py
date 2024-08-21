import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.signal import savgol_filter

# from physped.io.readers import read_preprocessed_trajectories_from_file
from physped.io.writers import save_trajectories
from physped.utils.functions import cartesian_to_polar_coordinates, compose_functions

log = logging.getLogger(__name__)


def rename_columns(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """Rename columns of a DataFrame.

    Args:
        df: The DataFrame to rename the columns of.
        colnames: A dictionary with the old column names as keys and the new column names as values.

    Returns:
        The DataFrame with the columns renamed.
    """
    # TODO : Use columnnames from parameters instead of renaming
    colnames = config.params.get("colnames", {})
    inverted_colnames = {v: k for k, v in colnames.items()}
    df.rename(columns=inverted_colnames, inplace=True)
    log.info("Columns renamed to %s", list(df.columns))
    return df


def cast_types(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """Cast the types of the columns of the DataFrame.

    Args:
        df: The DataFrame to cast the types of.
        config: The configuration object.

    Returns:
        The DataFrame with the columns casted to the correct types.
    """
    type_mapping = {
        "xf": float,
        "yf": float,
        "Pid": int,
        "time": int,
    }
    df = df.astype(type_mapping)
    log.info("Columns casted to %s", type_mapping)
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
        df: The DataFrame to add the trajectory step/observation to.
        config: The configuration object.

    Returns:
        The DataFrame with the trajectory step/observation added.
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
    filter with a polynomial order of 1.

    Args:
        df: The input DataFrame with the trajectories.

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


def transform_fast_velocity_to_polar_coordinates(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """Add columns with the velocity in polar coordinates to the DataFrame.

    Requires the columns 'uf' and 'vf' for the velocity in x and y direction, respectively.

    Args:
        df: The trajectory DataFrame to add polar coordinates to.

    Returns:
        The DataFrame with additional columns 'rf' and 'thetaf' for the fast velocity in polar coordinates.
    """
    df["rf"], df["thetaf"] = cartesian_to_polar_coordinates(df["uf"], df["vf"])
    log.info("Velocity in polar coordinates 'rf' and 'thetaf' added")
    return df


def transform_slow_velocity_to_polar_coordinates(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """Add columns with the velocity in polar coordinates to the DataFrame.

    Requires the columns 'us' and 'vs' for the velocity in x and y direction, respectively.

    Args:
        df: The trajectory DataFrame to add polar coordinates to.

    Returns:
        The DataFrame with additional columns 'rs' and 'thetas' for the slow velocity in polar coordinates.
    """
    df["rs"], df["thetas"] = cartesian_to_polar_coordinates(df["us"], df["vs"])
    log.info("Velocity in polar coordinates 'rs' and 'thetas' added")
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


def save_preprocessed_trajectories(df: pd.DataFrame, config: DictConfig) -> None:
    """Save the preprocessed trajectories to a file.

    The file is saved if the configuration object has the key 'save.preprocessed_trajectories' set to True.

    Args:
        df: The DataFrame with the preprocessed trajectories.
        config: The configuration object.

    Returns:
        The DataFrame with the preprocessed trajectories.
    """
    if config.save.preprocessed_trajectories:
        log.debug("Configuration 'save.preprocessed_trajectories' is set to True.")
        save_trajectories(df, Path.cwd().parent, config.filename.preprocessed_trajectories)
    return df


def preprocess_trajectories(trajectories: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """Preprocess trajectories.

    Args:
        trajectories: The DataFrame with the trajectories.

    Keyword Args:
        config: The configuration object.

    Returns:
        The preprocessed DataFrame with the trajectories.
    """
    preprocessing_pipeline = compose_functions(
        rename_columns,
        prune_short_trajectories,
        add_trajectory_index,
        compute_velocity_from_positions,
        transform_fast_velocity_to_polar_coordinates,
        save_preprocessed_trajectories,
    )
    return preprocessing_pipeline(trajectories, config=config)


# get_preprocessed_trajectories = {
#     "compute": preprocess_trajectories,
#     "read": read_preprocessed_trajectories_from_file,
# }
