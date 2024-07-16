import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from physped.io.readers import read_trajectories_from_path
from physped.io.writers import save_trajectories

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
            slow_path = integrate_slow_velocity_single_path(
                traj_i[kwargs["colname"]].iloc[0], traj_i[kwargs["vel_col"]], dt=kwargs["dt"]
            )
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


def process_slow_modes(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    dt = config.params.model["dt"]
    tauu = config.params.model["tauu"]
    window_length = config.params.minimum_trajectory_length
    slow_velocities_algorithm = config.params.model.slow_velocities_algorithm
    log.info("Slow velocity algorithm: %s", slow_velocities_algorithm)
    df["us"] = get_slow_algorithm(slow_velocities_algorithm)(df, colname="uf", tau=tauu, dt=dt, window_length=window_length)
    df["vs"] = get_slow_algorithm(slow_velocities_algorithm)(df, colname="vf", tau=tauu, dt=dt, window_length=window_length)
    df = add_velocity_in_polar_coordinates(df, mode="s")
    df["thetas"] = periodic_angular_conditions(df["thetas"], config.params.grid.bins["theta"])

    taux = config.params.model["taux"]
    slow_positions_algorithm = config.params.model.slow_positions_algorithm
    log.info("Slow position algorithm: %s", slow_positions_algorithm)
    df["xs"] = get_slow_algorithm(slow_positions_algorithm)(
        df, colname="xf", vel_col="us", tau=taux, dt=dt, window_length=window_length
    )
    df["ys"] = get_slow_algorithm(slow_positions_algorithm)(
        df, colname="yf", vel_col="vs", tau=taux, dt=dt, window_length=window_length
    )
    return df


def periodic_angular_conditions(theta, thetabins):
    theta -= thetabins[0]
    theta = theta % (2 * np.pi)
    theta += thetabins[0]
    return theta


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
            lambda x: savgol_filter(x, window_length=window_length, polyorder=1, deriv=1, mode="interp") * framerate
        )
    return df


def add_acceleration(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    framerate = parameters.fps
    groupby = "Pid"
    xcol = "uf"
    ycol = "vf"
    new_col = {"uf": "axf", "vf": "ayf"}
    window_length = parameters.velocity_window_length
    # window_length = parameters.minimum_trajectory_length - 1
    for direction in [xcol, ycol]:
        df.loc[:, new_col[direction]] = df.groupby([groupby])[direction].transform(
            lambda x: savgol_filter(x, window_length=window_length, polyorder=2, deriv=2, mode="interp") * framerate
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

    # axf = parameters.colnames.get("axf", None)
    # ayf = parameters.colnames.get("ayf", None)
    # if axf is None or ayf is None:
    #     log.warning("Columns for acceleration not found in parameters. Calculating acceleration.")
    #     df = add_acceleration(df, parameters)
    #     log.info("Acceleration added.")

    df = add_velocity_in_polar_coordinates(df, mode="f")
    df["thetaf"] = periodic_angular_conditions(df["thetaf"], config.params.grid.bins["theta"])
    log.info("Velocity transformed to polar coordinates.")

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
