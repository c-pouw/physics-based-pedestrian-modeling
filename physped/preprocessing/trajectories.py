import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from physped.io.readers import read_trajectories_from_path
from physped.io.writers import save_trajectories

log = logging.getLogger(__name__)


def compute_slow_modes_geert(xf: pd.Series, tau: float, dt: float) -> pd.Series:
    """Compute slow modes."""
    xfast = list(xf)
    xslow = list(xf)
    for i in range(len(xf) - 1):
        xslow[i + 1] = (1 - dt / tau) * xslow[i] + dt / tau * xfast[i]
    xs = xf.copy()
    xs.loc[:] = xslow
    return xs


def low_pass_filter(x: pd.Series, tau: float, dt: float) -> pd.Series:
    """
    Simple first-order low-pass filter.

    Args:
        x: Input signal (list or numpy array).
        tau: Time constant.
        dt: Time step.

    Returns:
        y: Filtered signal.
    """
    # y = [0] * len(x)  # Initialize output array
    y = copy.deepcopy(x)
    alpha = dt / (tau + dt)

    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]

    return y


def compute_slow_modes_numpy(xf: pd.Series, tau: float, dt: float) -> np.ndarray:
    """Compute slow modes."""
    # xf = list(xf)
    # xfast = xf.copy()
    xslow = list(xf)  # .copy()
    for i in range(len(xf) - 1):
        xslow[i + 1] = (1 - dt / tau) * xslow[i] + dt / tau * xslow[i]
    return xslow


def compute_slow_modes_ewm(xf: pd.Series, tau: float, dt: float) -> pd.Series:
    """
    Compute the slow modes of a time series using an exponential moving average filter.

    Parameters:
    - xf (pd.Series): The time series to compute the slow modes of.
    - tau (float): The time constant of the filter.
    - dt (float): The time step of the filter.

    Returns:
    - A new pandas Series object containing the slow modes of the input time series.
    """
    return xf.ewm(alpha=dt / tau, adjust=False).mean()


def compute_slow_modes_convolve(xf: np.ndarray, tau: float, dt: float) -> np.ndarray:
    """
    Compute the slow modes of a time series using an exponential moving average filter.

    Parameters:
    - xf (np.ndarray): The time series to compute the slow modes of.
    - tau (float): The time constant of the filter.
    - dt (float): The time step of the filter.

    Returns:
    - A new NumPy array containing the slow modes of the input time series.
    """
    # print(xf)
    # xf = list(xf)
    alpha = dt / tau
    weights = np.exp(-np.arange(len(xf)) * alpha)
    weights /= weights.sum()
    return np.convolve(xf, weights, mode="same")


def savgolfilter(xf: pd.Series, tau: float, dt: float) -> np.ndarray:
    window_length = min(len(xf) - 1, 40)
    return savgol_filter(xf, window_length, polyorder=1, deriv=0, mode="interp")


def compute_all_slow_modes(
    trajectories: pd.DataFrame,
    observables: list,
    tau: float,
    dt: float,
    slow_mode_algo=None,
) -> pd.DataFrame:
    """
    Compute the slow mode for all the observables.

    Parameters:
    - trajectories (pd.DataFrame): The DataFrame of trajectories.
    - observables (list): A list of observable names.
    - tau (float): The slow mode time constant.
    - dt (float): The time step.

    Returns:
    - The DataFrame with the slow modes added.
    """
    # compute_slow_modes = compute_slow_modes_new
    for obs in observables:
        trajectories[obs + "s"] = trajectories.groupby("Pid")[obs + "f"].transform(lambda x: slow_mode_algo(x, tau, dt))
    # log.info(f'Slow modes computed.')
    return trajectories


def add_velocity_in_polar_coordinates(df: pd.DataFrame, mode: str = "f") -> pd.DataFrame:
    """
    Add columns with velocity in polar coordinates to the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to add polar coordinates to.
    - mode (str): The mode to add polar coordinates for ('f' or 's').

    Returns:
    - The DataFrame with the 'r' and 'theta' columns added.
    """
    df["r" + mode] = np.hypot(df["u" + mode], df["v" + mode])
    df["theta" + mode] = np.arctan2(df["v" + mode], df["u" + mode])
    return df


def add_trajectory_step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Add a column with the observation/step numbers of each trajectory.

    Parameters:
    - df (pd.DataFrame): The DataFrame to add the trajectory step/observation to.

    Returns:
    - The DataFrame with the trajectory step/observation added.
    """
    # pid_col, time_col = params.colnames.Pid, params.colnames.time
    pid_col, time_col = "Pid", "time"
    df.sort_values(by=[pid_col, time_col], inplace=True)
    df["k"] = df.groupby(pid_col)[pid_col].transform(lambda x: np.arange(x.size))
    return df


def rename_columns(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Rename columns of a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to rename the columns of.
    - colnames (dict): A dictionary with the old column names as keys and the new column names as values.

    Returns:
    - The DataFrame with the columns renamed.
    """
    colnames = parameters.get("colnames", {})
    inverted_colnames = {v: k for k, v in colnames.items()}
    df.rename(columns=inverted_colnames, inplace=True)
    return df


def prune_short_trajectories(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    # df.reset_index(inplace=True, drop=True)
    df["traj_len"] = df.groupby(["Pid"])["Pid"].transform("size")
    df = df[df.traj_len > parameters.minimum_trajectory_length].copy()
    return df


def add_velocity(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Add velocity to dataframe.

    This function calculates the velocity of each pedestrian in the input DataFrame
    based on their position data. The velocity is calculated using the Savitzky-Golay
    filter with a window size of 49 and a polynomial order of 1.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the position data.
        groupby (str, optional): The column name to group the data by. Default is "Pid".
        xpos (str, optional): The column name for the x-position data. Default is "xf".
        ypos (str, optional): The column name for the y-position data. Default is "yf".

    Returns:
        pd.DataFrame: The input DataFrame with velocity columns added.

    Raises:
        ValueError: If the input DataFrame is empty or does not contain the specified columns.

    Example:
        >>> df = pd.DataFrame({'Pid': [1, 1, 2, 2], 'xf': [0, 1, 0, 1], 'yf': [0, 1, 0, 1]})
        >>> add_velocity(df)
           Pid  xf  yf   uf   vf
        0    1   0   0  0.0  0.0
        1    1   1   1  0.0  0.0
        2    2   0   0  0.0  0.0
        3    2   1   1  0.0  0.0
    """
    framerate = parameters.fps
    groupby = "Pid"
    xpos = "xf"
    ypos = "yf"
    pos_to_vel = {"xf": "uf", "yf": "vf"}
    window_length = parameters.velocity_window_length
    # window_length = parameters.minimum_trajectory_length - 1
    for direction in [xpos, ypos]:
        df.loc[:, pos_to_vel[direction]] = df.groupby([groupby])[direction].transform(
            lambda x: savgol_filter(x, window_length=window_length, polyorder=2, deriv=1, mode="interp") * framerate
        )
    return df


def preprocess_trajectories(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    parameters = config.params
    filepath = Path.cwd().parent / config.filename.preprocessed_trajectories

    # TODO : Move to separate function
    if config.read.preprocessed_trajectories:
        log.debug("Configuration 'read.preprocessed_trajectories' is set to True.")
        try:
            preprocessed_trajectories = read_trajectories_from_path(filepath)
            log.warning("Preprocessed trajectories read from file.")
            log.debug("Filepath %s", filepath.relative_to(config.root_dir))
            return preprocessed_trajectories
        except FileNotFoundError as e:
            log.error("Preprocessed trajectories not found: %s", e)

    log.info("Start preprocessing of the recorded trajectories.")
    # TODO : Use columnnames from parameters instead of renaming
    df = rename_columns(df, parameters)
    log.info("Columns renamed to %s", list(df.columns))
    df = prune_short_trajectories(df, parameters)
    log.info("Short trajectories with less than %s observations removed.", parameters.minimum_trajectory_length)
    df = add_trajectory_step(df, parameters)
    log.info("Trajectory step added.")

    uf = parameters.colnames.get("uf", None)
    vf = parameters.colnames.get("vf", None)
    if uf is None or vf is None:
        log.warning("Columns for velocity not found in parameters. Calculating velocity.")
        df = add_velocity(df, parameters)
        log.info("Velocity added.")
    df = add_velocity_in_polar_coordinates(df, mode="f")
    log.info("Velocity transformed to polar coordinates.")
    # TODO: Separate slow modes from preprocessing.
    slow_mode_algorithm = compute_slow_modes_geert
    log.info("Compute slow modes with %s.", slow_mode_algorithm.__name__)
    df = compute_all_slow_modes(
        df,
        ["x", "y", "u", "v"],  # , "r", "theta"],
        tau=parameters["taux"],
        dt=parameters["dt"],
        slow_mode_algo=slow_mode_algorithm,
    )
    df = add_velocity_in_polar_coordinates(df, mode="s")
    log.info("Slow mode velocity transformed to polar coordinates.")
    # if parameters.intermediate_save.preprocessed_trajectories:
    if config.save.preprocessed_trajectories:
        log.debug("Configuration 'save.preprocessed_trajectories' is set to True.")
        save_trajectories(df, Path.cwd().parent, config.filename.preprocessed_trajectories)
    return df


def filter_trajectories_by_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Filter trajectories by their average velocity."""
    log.info("Start filtering trajectories by velocity.")
    umin = 0.2
    df = df[df.groupby("Pid")["uf"].transform("mean") > umin]
    log.info("Trajectories filtered by velocity.")
    return df
