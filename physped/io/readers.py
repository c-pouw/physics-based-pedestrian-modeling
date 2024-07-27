"""Trajectory readers for the pathintegral code."""

import glob
import io
import logging
import pickle
import zipfile
from pathlib import Path

# from scipy.signal import savgol_filter
import pandas as pd
import requests
from scipy import signal
from tqdm import tqdm

from physped.core.piecewise_potential import PiecewisePotential

log = logging.getLogger(__name__)


def read_trajectories_from_path(filepath: Path) -> pd.DataFrame:
    """Read trajectories from file. Used to read intermediate results."""
    return pd.read_csv(filepath)


def read_piecewise_potential_from_file(filepath: Path) -> PiecewisePotential:
    """
    Reads a piecewise potential from a file using pickle.

    :param filename: The path to the file containing the piecewise potential.
    :type filename: str
    :return: The piecewise potential.
    """
    with open(filepath, "rb") as f:
        val = pickle.load(f)
    return val


def read_narrow_corridor_paths_local(config) -> pd.DataFrame:
    trajectory_data_dir = Path(config.trajectory_data_dir)
    log.info("Start reading single paths data set.")
    archive = zipfile.ZipFile(trajectory_data_dir / "data.zip")

    with archive.open("left-to-right.ssv") as f:
        paths_ltr = f.read().decode("utf-8")

    with archive.open("right-to-left.ssv") as f:
        paths_rtl = f.read().decode("utf-8")
    return paths_ltr, paths_rtl


def read_narrow_corridor_paths_4tu(config) -> pd.DataFrame:
    link = "https://data.4tu.nl/ndownloader/items/b8e30f8c-3931-4604-842a-77c7fb8ac3fc/versions/1"
    bytestring = requests.get(link, timeout=10)
    with zipfile.ZipFile(io.BytesIO(bytestring.content), "r") as outerzip:
        with zipfile.ZipFile(outerzip.open("data.zip")) as innerzip:
            with innerzip.open("left-to-right.ssv") as paths_ltr:
                paths_ltr = paths_ltr.read().decode("utf-8")
            with innerzip.open("right-to-left.ssv") as paths_rtl:
                paths_rtl = paths_rtl.read().decode("utf-8")
    return paths_ltr, paths_rtl


narrow_corridor_path_reader = {
    "local": read_narrow_corridor_paths_local,
    "4tu": read_narrow_corridor_paths_4tu,
}


def read_single_paths(config) -> pd.DataFrame:
    """Read the single paths data set."""
    data_source = config.params.data_source
    paths_ltr, paths_rtl = narrow_corridor_path_reader[data_source](config)

    df1 = pd.read_csv(io.StringIO(paths_ltr), sep=" ")
    df2 = pd.read_csv(io.StringIO(paths_rtl), sep=" ")
    df = pd.concat([df1, df2], ignore_index=True)

    df["X_SG"] = df["X_SG"] + 0.1
    df["Y_SG"] = df["Y_SG"] - 0.05
    df.rename(columns={"X_SG": "xf", "Y_SG": "yf", "U_SG": "uf", "V_SG": "vf"}, inplace=True)
    log.info("Finished reading single paths data set.")
    return df


def read_parallel_paths(config) -> pd.DataFrame:
    """Read the parallel paths data set."""
    trajectory_data_dir = Path(config.trajectory_data_dir)
    file_path = trajectory_data_dir / "df_single_pedestrians_small.h5"
    df = pd.read_hdf(file_path)
    df.rename(columns={"X_SG": "xf", "Y_SG": "yf", "U_SG": "uf", "V_SG": "vf"}, inplace=True)
    df["xf"] = df["xf"] - 0.4
    df["yf"] = df["yf"] + 0.3
    df["Pid"] = df.groupby(["Pid", "day_id"]).ngroup()
    df.reset_index(inplace=True)
    df = df.query("Umean>0.5").loc[abs(df.X0 - df.X1) > 2]
    df = df.groupby("Pid").filter(lambda x: max(x.uf) < 3.5)
    # df["time"] = df["frame"]
    return df


# def read_intersecting_paths_synthetic(config) -> pd.DataFrame:
#     """Read the intersecting paths data set."""
#     trajectory_data_dir = Path(config.trajectory_data_dir)
#     file_path = trajectory_data_dir / "simulations_crossing.parquet"
#     df = pd.read_parquet(file_path)
#     df.rename(columns={"X_SG": "xf", "Y_SG": "yf", "U_SG": "uf", "V_SG": "vf"}, inplace=True)
#     df["k"] = df.groupby("Pid").cumcount()
#     df["time"] = df["k"]
#     return df


def read_intersecting_paths(config) -> pd.DataFrame:
    data_source = config.params.data_source
    paths_ltr, paths_rtl = narrow_corridor_path_reader[data_source](config)

    df1 = pd.read_csv(io.StringIO(paths_ltr), sep=" ")
    df1["X_SG"] = df1["X_SG"] + 0.1
    df1["Y_SG"] = df1["Y_SG"] - 0.05

    df2 = pd.read_csv(io.StringIO(paths_rtl), sep=" ")
    df2["X_SG"] = df2["X_SG"] + 0.1
    df2["Y_SG"] = df2["Y_SG"] - 0.05
    df2.rename(  # swap x and y coordinates
        columns={"X": "Y", "Y": "X", "X_SG": "Y_SG", "Y_SG": "X_SG", "U_SG": "V_SG", "V_SG": "U_SG"}, inplace=True
    )

    df = pd.concat([df1, df2], ignore_index=True)

    log.info("Finished reading single paths data set.")
    return df


def read_curved_paths_synthetic(config) -> pd.DataFrame:
    """Read the curved paths data set."""
    trajectory_data_dir = Path(config.trajectory_data_dir)
    file_path = trajectory_data_dir / "artificial_measurements_ellipse.parquet"
    df = pd.read_parquet(file_path)
    df = df.rename(columns={"x": "xf", "y": "yf", "xdot": "uf", "ydot": "vf"})
    return df


def filter_trajectory(df, cutoff=0.16, order=4):
    b, a = signal.butter(order, cutoff, "low")
    df = df.sort_values(["particle", "time"])
    df = df.groupby("particle").filter(lambda x: len(x) > 52)

    f_df = df.groupby(df["particle"]).apply(lambda x: pd.DataFrame(signal.filtfilt(b, a, x[["x", "y"]].values, axis=0)))
    df[["x", "y"]] = f_df.set_index(df.index)
    return df


# def savgol_smoothing(df: pd.DataFrame, smooth_colname, groupby_colname="Pid"):
#     slow = df.groupby(groupby_colname)[smooth_colname].transform(
#         lambda x: signal.savgol_filter(x, window_length=9, polyorder=1, deriv=0, mode="interp")
#     )
#     return slow


# def read_curved_paths(config) -> pd.DataFrame:
#     # Trajectories recorded during experiment 1
#     trajectory_data_dir = Path(config.trajectory_data_dir)
#     trajs = pd.read_csv(trajectory_data_dir / "linecross-1.csv")
#     # trajs = filter_trajectory(trajs)
#     pid_column = "particle"
#     time_column = "time"

#     conversion_X = 2.30405921919033
#     conversion_Y = 2.35579871138595
#     trajs = trajs[trajs.frame > 380].copy()
#     trajs["x"] = trajs["x"] - np.mean(trajs["x"]) - 30
#     trajs["y"] = trajs["y"] - np.mean(trajs["y"]) + 35
#     trajs1 = trajs.copy()

#     # Trajectories recorded during experiment 2
#     trajs = pd.read_csv(trajectory_data_dir / "linecross-2.csv")
#     trajs = filter_trajectory(trajs)
#     trajs = trajs[trajs.frame > 380].copy()
#     trajs["x"] = trajs["x"] - np.mean(trajs["x"]) + 17
#     trajs["y"] = trajs["y"] - np.mean(trajs["y"]) + 33
#     trajs[pid_column] += np.max(trajs1[pid_column])
#     trajs2 = trajs.copy()

#     trajs = pd.concat([trajs1, trajs2])
#     trajs["x"] = trajs["x"] * conversion_X / 100
#     trajs["y"] = trajs["y"] * conversion_Y / 100

#     # trajs["v_x_m"] = trajs["v_x_m"].replace(-99, np.nan).interpolate()
#     # trajs["v_y_m"] = trajs["v_y_m"].replace(-99, np.nan).interpolate()

#     trajs["traj_len"] = trajs.groupby([pid_column])[pid_column].transform("size")
#     trajs = trajs[trajs.traj_len > 10].copy()
#     trajs.sort_values(by=[pid_column, time_column], inplace=True)
#     trajs["k"] = trajs.groupby(pid_column)[pid_column].transform(lambda x: np.arange(x.size))
#     # trajs["kp"] = trajs["k"] // 320
#     # trajs["new_pid"] = trajs.apply(lambda x: int(f"{x[pid_column]:06}{x['kp']:04}"), axis=1)
#     # trajs["new_pid"] = trajs[pid_column] * 100000 + trajs["kp"] + 100
#     trajs["x"] = savgol_smoothing(trajs, "x", pid_column)  # Smooth the noisy trajectories to get reasonable velocities
#     trajs["y"] = savgol_smoothing(trajs, "y", pid_column)
#     return trajs


def read_ehv_pf34_paths_geert(config) -> pd.DataFrame:
    """Read the station paths data set."""
    trajectory_data_dir = Path(config.trajectory_data_dir)
    file_path = trajectory_data_dir / "trajectories_EHV_platform_2_1_refined.parquet"
    df = pd.read_parquet(file_path)
    # df = df[["date_time_utc", "Pid", "xf", "yf"]]

    # Rotate the domain
    df.rename({"xf": "yf", "yf": "xf", "uf": "vf", "vf": "uf"}, axis=1, inplace=True)
    return df


def filter_part_of_the_domain(df, xmin, xmax):
    df = df[df["x_pos"] > xmin].copy()
    df = df[df["x_pos"] < xmax].copy()
    return df


def read_ehv_pf34_paths_local(config) -> pd.DataFrame:
    trajectory_data_dir = Path(config.trajectory_data_dir)
    glob_string = f"{str(trajectory_data_dir)}/ehv_pf34/*.parquet"
    filelist = glob.glob(glob_string)
    df_list = []
    for file in tqdm(filelist, ascii=True):
        df = pd.read_parquet(file)
        df_list.append(df)

    df = pd.concat(df_list)

    # Rotate the domain
    df.rename({"x_pos": "y_pos", "y_pos": "x_pos"}, axis=1, inplace=True)

    # Convert position units to meters
    df["x_pos"] /= 1000
    df["y_pos"] /= 1000

    df = filter_part_of_the_domain(df, xmin=50, xmax=70)
    return df


ehv_pf34_path_reader = {
    "geert": read_ehv_pf34_paths_geert,
    "local": read_ehv_pf34_paths_local,
}


def read_station_paths(config) -> pd.DataFrame:
    """Read the station paths data set."""
    path_reader = ehv_pf34_path_reader[config.params.data_source]
    df = path_reader(config)
    return df


def read_asdz_pf34_paths(config) -> pd.DataFrame:
    # TODO: Make separate readers for local and 4tu data sources
    if config.params.data_source == "local":
        trajectory_data_dir = Path(config.trajectory_data_dir)
        file_path = trajectory_data_dir / "Amsterdam Zuid - platform 3-4 - set1.csv"
        df = pd.read_csv(file_path)
    elif config.params.data_source == "4tu":
        link = "https://data.4tu.nl/file/7d78a5e3-6142-49fe-be03-e4c707322863/40ea5cd9-95dc-4e3c-8760-7f4dd543eae7"
        bytestring = requests.get(link, timeout=10)

        with zipfile.ZipFile(io.BytesIO(bytestring.content), "r") as zipped_file:
            with zipped_file.open("Amsterdam Zuid - platform 3-4 - set1.csv") as paths:
                paths = paths.read().decode("utf-8")

        df = pd.read_csv(io.StringIO(paths), sep=",")

    df["x_pos"] /= 1000
    df["y_pos"] /= 1000
    return df


def read_utrecht_pf5_paths_4tu(config):
    link = "https://data.4tu.nl/file/d4d548c6-d198-49b3-986c-e22319970a5e/a58041fb-0318-4bee-9b2c-934bd8e5df83"
    bytestring = requests.get(link, timeout=10)

    with zipfile.ZipFile(io.BytesIO(bytestring.content), "r") as zipped_file:
        with zipped_file.open("Utrecht Centraal - platform 5 - set99.csv") as paths:
            paths = paths.read().decode("utf-8")

    df = pd.read_csv(io.StringIO(paths), sep=",")
    return df


def read_utrecht_pf5_paths_local(config):
    file_list = glob.glob(config.trajectory_data_dir + "/Utrecht*.csv")
    file_path = file_list[0]
    return pd.read_csv(file_path)


utrecht_pf5_path_reader = {
    "local": read_utrecht_pf5_paths_local,
    "4tu": read_utrecht_pf5_paths_4tu,
}


def read_utrecht_pf5_paths(config) -> pd.DataFrame:
    path_reader = utrecht_pf5_path_reader[config.params.data_source]
    df = path_reader(config)
    df["x_pos"] /= 1000  # Convert milimeters to meters
    df["y_pos"] /= 1000
    return df


def read_asdz_pf12_paths_4tu(config):
    link = "https://data.4tu.nl/file/af4ef093-69ef-4e1c-8fbc-c40c447c618c/d07747f0-9101-4cfc-9939-4a63c2677b22"
    bytestring = requests.get(link, timeout=10)

    with zipfile.ZipFile(io.BytesIO(bytestring.content), "r") as zipped_file:
        with zipped_file.open("Amsterdam Zuid - platform 1-2 - set32.csv") as paths:
            paths = paths.read().decode("utf-8")

    df = pd.read_csv(io.StringIO(paths), sep=",")
    return df


def read_asdz_pf12_paths_local(config):
    file_list = glob.glob(config.trajectory_data_dir + "/Amsterdam*Zuid*1-2*.csv")
    file_path = file_list[0]
    return pd.read_csv(file_path)


asdz_pf12_path_reader = {
    "local": read_asdz_pf12_paths_local,
    "4tu": read_asdz_pf12_paths_4tu,
}


def read_asdz_pf12_paths(config) -> pd.DataFrame:
    path_reader = asdz_pf12_path_reader[config.params.data_source]
    df = path_reader(config)
    df["x_pos"] /= 1000  # Convert milimeters to meters
    df["y_pos"] /= 1000
    return df


trajectory_reader = {
    "single_paths": read_single_paths,
    "parallel_paths": read_parallel_paths,
    "intersecting_paths": read_intersecting_paths,
    # "intersecting_paths_synthetic": read_intersecting_paths_synthetic,
    # "curved_paths": read_curved_paths,
    "curved_paths_synthetic": read_curved_paths_synthetic,
    "station_paths": read_station_paths,
    "asdz_pf34": read_asdz_pf34_paths,
    "utrecht_pf5": read_utrecht_pf5_paths,
    "asdz_pf12": read_asdz_pf12_paths,
}
