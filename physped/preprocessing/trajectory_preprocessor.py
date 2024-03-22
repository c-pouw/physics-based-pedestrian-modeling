import pandas as pd
import numpy as np
import logging

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


def compute_all_slow_modes(
    trajectories: pd.DataFrame,
    observables: list,
    tau: float,
    dt: float,
    slow_mode_algo=compute_slow_modes_numpy,
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
        trajectories[obs + "s"] = trajectories.groupby("Pid")[obs + "f"].transform(
            lambda x: slow_mode_algo(x, tau, dt)
        )
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
    colnames = params.get("colnames", {})
    pid_col = colnames.get("Pid", "Pid")
    time_col = colnames.get("time", "time")
    df.sort_values(by=[pid_col, time_col], inplace=True)
    df["k"] = df.groupby(pid_col)[pid_col].transform(lambda x: np.arange(x.size))
    return df


def preprocess_trajectories(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    log.info("Start trajectory preprocessing.")
    # print(df.columns)
    df = add_trajectory_step(df, parameters)  # , colnames={"Pid": "Pid", "time": "time"})
    log.info("Trajectory step added.")
    df = add_velocity_in_polar_coordinates(df, mode="f")
    log.info("Polar coordinates added.")
    df = compute_all_slow_modes(
        df,
        ["x", "y", "u", "v"],  # , "r", "theta"],
        tau=parameters["taux"],
        dt=parameters["dt"],
        slow_mode_algo=compute_slow_modes_geert,
    )
    df = add_velocity_in_polar_coordinates(df, mode="s")
    log.info("Slow modes computed.")
    log.info("Trajectories preprocessed.")
    # df = dg.convert_slow_velocities_to_polar(df)
    return df


def filter_trajectories_by_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Filter trajectories by mean velocity."""
    log.info("Start filtering trajectories by velocity.")
    umin = 0.2
    df = df[df.groupby("Pid")["uf"].transform("mean") > umin]
    log.info("Trajectories filtered by velocity.")
    return df
