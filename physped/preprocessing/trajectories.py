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

# def compute_slow_modes_with_shift(x: pd.Series, tau: float, dt: float) -> list:
#     x = list(x)
#     alpha = dt / tau
#     k = int(tau / dt)
#     # k = int(tau_x / (2 * delta_t))
#     x = x + [x[-1]] * (k-1)
#     x = x[k:]
#     xs = [x[0]]  # Initialize xs(0) to x(0)
#     for i in range(0, len(x)):
#         xs.append((1 - alpha) * xs[-1] + alpha * x[i])
#     return xs

# def compute_slow_modes_with_shift(x: pd.Series, tau: float, dt: float) -> list:
#     x = list(x)
#     alpha = dt / tau
#     k = int(tau / dt)
#     # k = int(tau_x / (2 * delta_t))
#     x = x + [x[-1]] * (k-1)
#     x = x[k:]
#     xs = [x[0]]  # Initialize xs(0) to x(0)
#     for i in range(0, len(x)):
#         xs.append((1 - alpha) * xs[-1] + alpha * x[i])
#     return xs


# def compute_slow_position(x, us, tau_x, delta_t):
#     x = list(x)
#     us = list(us)
#     alpha = delta_t / tau_x
#     xs = np.zeros_like(x)
#     xs[0] = x[0]  # Initialize xs(0) to x(0)
#     for i in range(0, len(x)-1):
#         xs[i+1] = (1 - alpha) * xs[i] + alpha * x[i] + delta_t * us[i]
#     return xs

# def compute_slow_velocity(u, accs, tau_x, delta_t):
#     u = list(u)
#     accs = list(accs)
#     alpha = delta_t / tau_x
#     us = np.zeros_like(u)
#     # us[0] = u[0]  # Initialize xs(0) to x(0)
#     us[0] = np.mean(u[:20])
#     for i in range(0, len(u)-1):
#         us[i+1] = (1 - alpha) * us[i] + alpha * u[i] #+ delta_t * accs[i]
#     return us


# def savgolfilter(xf: pd.Series, tau: float, dt: float) -> np.ndarray:
#     window_length = min(len(xf) - 1, 40)
#     return savgol_filter(xf, window_length, polyorder=1, deriv=0, mode="interp")


# def compute_all_slow_modes(trajectories, tau, dt):
#     # TODO: Improve this function
#     test_trajs_with_slow_modes = []
#     with logging_redirect_tqdm():
#         # for X_0 in tqdm(origins[:, :8], desc="Simulating trajectories", unit="trajs", total=origins.shape[0], miniters=10):
#         for _, traj_i in tqdm(trajectories.groupby('Pid'), desc = 'Computing slow modes', unit = 'trajs',
# total = len(trajectories.Pid.unique()), miniters = 100):
#             traj_i['us'] = compute_slow_velocity(traj_i['uf'], traj_i['axf'], tau, dt)
#             traj_i['vs'] = compute_slow_velocity(traj_i['vf'], traj_i['ayf'], tau, dt)
#             traj_i['xs'] = compute_slow_position(traj_i['xf'], traj_i['us'], tau, dt)
#             traj_i['ys'] = compute_slow_position(traj_i['yf'], traj_i['vs'], tau, dt)
#             test_trajs_with_slow_modes.append(traj_i)

#     test_trajs_with_slow_modes = pd.concat(test_trajs_with_slow_modes)
#     test_trajs_with_slow_modes.sort_values(by='time', inplace=True)
#     return test_trajs_with_slow_modes


# def compute_all_slow_modes_without_time_recentering(
#     trajectories: pd.DataFrame,
#     observables: list,
#     tau: float,
#     dt: float,
#     slow_mode_algo=None,
# ) -> pd.DataFrame:
#     """
#     Compute the slow mode for all the observables.

#     Parameters:
#     - trajectories (pd.DataFrame): The DataFrame of trajectories.
#     - observables (list): A list of observable names.
#     - tau (float): The slow mode time constant.
#     - dt (float): The time step.

#     Returns:
#     - The DataFrame with the slow modes added.
#     """
#     # compute_slow_modes = compute_slow_modes_new
#     for obs in observables:
#         trajectories[obs + "s"] = trajectories.groupby("Pid")[obs + "f"].transform(lambda x: slow_mode_algo(x, tau, dt))
#     # log.info(f'Slow modes computed.')
#     return trajectories


# def compute_all_slow_modes_geert(trajectories, taux, tauu, dt):
#     # TODO: Improve this function
#     test_trajs_with_slow_modes = []
#     with logging_redirect_tqdm():
#         for _, traj_i in tqdm(
#             trajectories.groupby("Pid"),
#             desc="Computing slow modes",
#             unit="trajs",
#             total=len(trajectories.Pid.unique()),
#             miniters=100,
#         ):

#             xf = traj_i["xf"]
#             yf = traj_i["yf"]
#             uf = traj_i["uf"]
#             vf = traj_i["vf"]

#             traj_i["us"] = compute_slow_modes_geert(uf, tauu, dt)
#             traj_i["vs"] = compute_slow_modes_geert(vf, tauu, dt)
#             # traj_i["xs"] = compute_slow_modes_geert(xf, taux, dt)
#             # traj_i["ys"] = compute_slow_modes_geert(yf, taux, dt)
#             traj_i["xs"] = xf
#             traj_i["ys"] = yf
#             test_trajs_with_slow_modes.append(traj_i)

#     test_trajs_with_slow_modes = pd.concat(test_trajs_with_slow_modes)
#     test_trajs_with_slow_modes.sort_values(by="time", inplace=True)
#     return test_trajs_with_slow_modes


# def compute_all_slow_modes_cas(trajectories, taux, tauu, dt):
#     # TODO: Improve this function
#     test_trajs_with_slow_modes = []
#     with logging_redirect_tqdm():
#         for _, traj_i in tqdm(
#             trajectories.groupby("Pid"),
#             desc="Computing slow modes",
#             unit="trajs",
#             total=len(trajectories.Pid.unique()),
#             miniters=100,
#         ):

#             xf = list(traj_i["xf"])
#             yf = list(traj_i["yf"])
#             uf = list(traj_i["uf"])
#             vf = list(traj_i["vf"])

#             usdot = np.zeros(len(traj_i))
#             vsdot = np.zeros(len(traj_i))
#             us0 = uf[0]  # init with the first value of the fast mode
#             vs0 = vf[0]
#             # us0 = uf[5]
#             # vs0 = vf[5]
#             # us0 = np.mean(uf[:20])
#             # vs0 = np.mean(vf[:20])

#             traj_i["us"] = compute_slow_dynamics(uf, usdot, us0, tauu, dt)
#             traj_i["vs"] = compute_slow_dynamics(vf, vsdot, vs0, tauu, dt)

#             xsdot = list(traj_i["us"])
#             ysdot = list(traj_i["vs"])
#             xs0 = xf[0]
#             ys0 = yf[0]

#             traj_i["xs"] = compute_slow_dynamics(xf, xsdot, xs0, taux, dt)
#             traj_i["ys"] = compute_slow_dynamics(yf, ysdot, ys0, taux, dt)
#             test_trajs_with_slow_modes.append(traj_i)

#     test_trajs_with_slow_modes = pd.concat(test_trajs_with_slow_modes)
#     test_trajs_with_slow_modes.sort_values(by="time", inplace=True)
#     return test_trajs_with_slow_modes


# def compute_velocity_slow_modes(df):
#     df[obs + "s"] = df.groupby("Pid")[obs + "f"].transform(lambda x: slow_mode_algo(x, tau, dt))
#     pass

# def compute_position_slow_modes():
#     pass


# def compute_slow_dynamics(x, xsdot, xs0, tau, delta_t):
#     if tau == 0:
#         alpha = 0
#     else:
#         alpha = delta_t / tau
#     xs = np.zeros_like(x)
#     xs[0] = xs0
#     for i in range(0, len(x) - 1):
#         xs[i + 1] = delta_t * xsdot[i] + (1 - alpha) * xs[i] + alpha * x[i]
#     return xs


# def compute_slow_modes_geert(xf: pd.Series, tau: float, dt: float) -> pd.Series:
#     """Compute slow modes."""
#     xfast = list(xf)
#     xslow = list(xf)
#     for i in range(len(xf) - 1):
#         xslow[i + 1] = (1 - dt / tau) * xslow[i] + dt / tau * xfast[i]
#     xs = xf.copy()
#     xs.loc[:] = xslow
#     return xs


def compute_slow_position_from_slow_velocity_single_path(x0, us, dt):
    # x = list(x)
    us = list(us)
    xs = np.zeros_like(us)
    xs[0] = x0  # Initialize xs(0) to x(0)
    for i in range(0, len(xs) - 1):
        xs[i + 1] = xs[i] + dt * us[i]
    return xs


def low_pass_filter_single_path(fast: pd.Series, tau: float, dt: float) -> pd.Series:
    """Compute slow dynamics of single trajectory with low pass filter."""
    fast = list(fast)
    slow = np.zeros_like(fast)
    slow[0] = fast[0]
    for i in range(len(fast) - 1):
        slow[i + 1] = (1 - dt / tau) * slow[i] + dt / tau * fast[i]
    return slow


SLOW_ALGORITHMS = {}


def register_slow_algorithm(name):
    def decorator(fn):
        SLOW_ALGORITHMS[name] = fn
        return fn

    return decorator


@register_slow_algorithm("low_pass_filter")
def low_pass_filter_all_paths(df, **kwargs):
    grouped_paths = df.groupby("Pid")[kwargs["colname"]]
    return grouped_paths.transform(lambda x: low_pass_filter_single_path(x, kwargs["tau"], kwargs["dt"]))


@register_slow_algorithm("use_fast_dynamics")
def use_fast_dynamics(df, **kwargs):
    return df[kwargs["colname"]]


@register_slow_algorithm("slow_position_from_slow_velocity")
def slow_position_from_slow_velocity(df, **kwargs):
    slow = []
    with logging_redirect_tqdm():
        for _, traj_i in tqdm(
            df.groupby("Pid"),
            desc=f"Computing {kwargs['colname'][0]}s with {kwargs['vel_col']}",
            unit="trajs",
            total=len(df.Pid.unique()),
            miniters=100,
        ):
            slow_path = compute_slow_position_from_slow_velocity_single_path(
                traj_i[kwargs["colname"]].iloc[0], traj_i[kwargs["vel_col"]], dt=kwargs["dt"]
            )
            slow.extend(slow_path)
    return slow


def get_slow_algorithm(name):
    return SLOW_ALGORITHMS.get(name)


def process_slow_modes(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    dt = config.params.model["dt"]
    tauu = config.params.model["tauu"]
    slow_velocities_algorithm = config.params.model.slow_velocities_algorithm
    log.info("Slow velocity algorithm: %s", slow_velocities_algorithm)
    df["us"] = get_slow_algorithm(slow_velocities_algorithm)(df, colname="uf", tau=tauu, dt=dt)
    df["vs"] = get_slow_algorithm(slow_velocities_algorithm)(df, colname="vf", tau=tauu, dt=dt)
    df = add_velocity_in_polar_coordinates(df, mode="s")

    taux = config.params.model["taux"]
    slow_positions_algorithm = config.params.model.slow_positions_algorithm
    log.info("Slow position algorithm: %s", slow_positions_algorithm)
    df["xs"] = get_slow_algorithm(slow_positions_algorithm)(df, colname="xf", vel_col="us", tau=taux, dt=dt)
    df["ys"] = get_slow_algorithm(slow_positions_algorithm)(df, colname="yf", vel_col="vs", tau=taux, dt=dt)
    return df


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
