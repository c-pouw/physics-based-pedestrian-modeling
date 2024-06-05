"""Trajectory readers for the pathintegral code."""

import io
import logging
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import signal
from tqdm import tqdm

from physped.core.piecewise_potential import PiecewisePotential

log = logging.getLogger(__name__)


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


# def read_minimal_dataset_for_testing(config) -> pd.DataFrame:
#     """Read the single paths data set."""
#     log.info("Start reading single paths data set.")
#     trajectory_data_dir = Path(config.trajectory_data_dir)
#     with ZipFile(trajectory_data_dir / "minimal_test_dataset.zip", "r") as archive:
#         with archive.open("single_paths_rtl.csv") as f:
#             paths = pd.read_csv(f)

#     log.info("Finished reading single paths data set.")
#     return paths


def read_single_paths(config) -> pd.DataFrame:
    """Read the single paths data set."""
    if config.params.data_source == "local":
        trajectory_data_dir = Path(config.trajectory_data_dir)
        log.info("Start reading single paths data set.")
        archive = zipfile.ZipFile(trajectory_data_dir / "data.zip")

        with archive.open("left-to-right.ssv") as f:
            paths_ltr = f.read().decode("utf-8")

        with archive.open("right-to-left.ssv") as f:
            paths_rtl = f.read().decode("utf-8")

    elif config.params.data_source == "4tu":
        link = "https://data.4tu.nl/ndownloader/items/b8e30f8c-3931-4604-842a-77c7fb8ac3fc/versions/1"
        bytestring = requests.get(link, timeout=10)
        with zipfile.ZipFile(io.BytesIO(bytestring.content), "r") as outerzip:
            with zipfile.ZipFile(outerzip.open("data.zip")) as innerzip:
                with innerzip.open("left-to-right.ssv") as paths_ltr:
                    paths_ltr = paths_ltr.read().decode("utf-8")
                with innerzip.open("right-to-left.ssv") as paths_rtl:
                    paths_rtl = paths_rtl.read().decode("utf-8")

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


def read_intersecting_paths_synthetic(config) -> pd.DataFrame:
    """Read the intersecting paths data set."""
    trajectory_data_dir = Path(config.trajectory_data_dir)
    file_path = trajectory_data_dir / "simulations_crossing.parquet"
    df = pd.read_parquet(file_path)
    df.rename(columns={"X_SG": "xf", "Y_SG": "yf", "U_SG": "uf", "V_SG": "vf"}, inplace=True)
    df["k"] = df.groupby("Pid").cumcount()
    df["time"] = df["k"]
    return df


def read_intersecting_paths(config) -> pd.DataFrame:
    link = "https://data.4tu.nl/ndownloader/items/b8e30f8c-3931-4604-842a-77c7fb8ac3fc/versions/1"
    bytestring = requests.get(link, timeout=10)
    with zipfile.ZipFile(io.BytesIO(bytestring.content), "r") as outerzip:
        with zipfile.ZipFile(outerzip.open("data.zip")) as innerzip:
            with innerzip.open("left-to-right.ssv") as paths_ltr:
                paths_ltr = paths_ltr.read().decode("utf-8")
            with innerzip.open("right-to-left.ssv") as paths_rtl:
                paths_rtl = paths_rtl.read().decode("utf-8")

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

    # df2 = pd.read_csv(io.StringIO(data_str), sep=" ")
    # df = pd.concat([df1, df2], ignore_index=True)
    # df["X_SG"] = df["X_SG"] + 0.1
    # df["Y_SG"] = df["Y_SG"] - 0.05
    # df.rename(columns={"X_SG": "xf", "Y_SG": "yf", "U_SG": "uf", "V_SG": "vf"}, inplace=True)
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


def read_curved_paths(config) -> pd.DataFrame:
    # trajectory_data_dir = Path(config.trajectory_data_dir)
    # trajs = pd.read_csv(trajectory_data_dir / "linecross-2.csv")
    # pid_column = "particle"
    # # x_column = 'x'
    # # y_column = 'y'
    # time_column = "time"
    # conversion_X = 2.30405921919033
    # conversion_Y = 2.35579871138595
    # trajs = trajs[trajs.frame > 380].copy()
    # trajs["x"] = trajs["x"] - np.mean(trajs["x"]) - 30
    # trajs["y"] = trajs["y"] - np.mean(trajs["y"]) + 35
    # trajs["x"] = trajs["x"] * conversion_X / 100
    # trajs["y"] = trajs["y"] * conversion_Y / 100

    # Trajectories recorded during experiment 1
    trajectory_data_dir = Path(config.trajectory_data_dir)
    trajs = pd.read_csv(trajectory_data_dir / "linecross-1.csv")
    trajs = filter_trajectory(trajs)
    pid_column = "particle"
    time_column = "time"
    conversion_X = 2.30405921919033
    conversion_Y = 2.35579871138595
    trajs = trajs[trajs.frame > 380].copy()
    trajs["x"] = trajs["x"] - np.mean(trajs["x"]) - 30
    trajs["y"] = trajs["y"] - np.mean(trajs["y"]) + 35
    trajs1 = trajs.copy()

    # Trajectories recorded during experiment 2
    trajs = pd.read_csv(trajectory_data_dir / "linecross-2.csv")
    trajs = filter_trajectory(trajs)
    trajs = trajs[trajs.frame > 380].copy()
    trajs["x"] = trajs["x"] - np.mean(trajs["x"]) + 17
    trajs["y"] = trajs["y"] - np.mean(trajs["y"]) + 33
    trajs[pid_column] += np.max(trajs1[pid_column])
    trajs2 = trajs.copy()

    trajs = pd.concat([trajs1, trajs2])
    trajs["x"] = trajs["x"] * conversion_X / 100
    trajs["y"] = trajs["y"] * conversion_Y / 100

    trajs["v_x_m"] = trajs["v_x_m"].replace(-99, np.nan).interpolate()
    trajs["v_y_m"] = trajs["v_y_m"].replace(-99, np.nan).interpolate()

    trajs["traj_len"] = trajs.groupby([pid_column])[pid_column].transform("size")
    trajs = trajs[trajs.traj_len > 10].copy()
    trajs.sort_values(by=[pid_column, time_column], inplace=True)
    trajs["k"] = trajs.groupby(pid_column)[pid_column].transform(lambda x: np.arange(x.size))
    trajs["kp"] = trajs["k"] // 320
    trajs["new_pid"] = trajs.apply(lambda x: int(f"{x[pid_column]:06}{x['kp']:04}"), axis=1)
    # trajs["new_pid"] = trajs[pid_column] * 100000 + trajs["kp"] + 100
    return trajs


def read_station_paths(config) -> pd.DataFrame:
    """Read the station paths data set."""
    trajectory_data_dir = Path(config.trajectory_data_dir)
    file_path = trajectory_data_dir / "trajectories_EHV_platform_2_1_refined.parquet"
    df = pd.read_parquet(file_path)
    df = df[["date_time_utc", "Pid", "xf", "yf"]]
    df.rename({"xf": "yf", "yf": "xf", "uf": "vf", "vf": "uf"}, axis=1, inplace=True)
    return df


def read_asdz_pf34(config) -> pd.DataFrame:
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


def read_ehv_train_station_multifile(filelist) -> pd.DataFrame:
    df_list = []
    for file in tqdm(filelist, ascii=True):
        df = pd.read_parquet(file)
        df_list.append(df)

    df = pd.concat(df_list)
    df = preprocess_ehv(df)
    return df


def preprocess_ehv(df: pd.DataFrame) -> pd.DataFrame:
    for col_name in ["x_pos", "y_pos"]:
        df[col_name] = df[col_name].astype(float)
    df.reset_index(inplace=True, drop=True)
    df["traj_len"] = df.groupby(["tracked_object"])["tracked_object"].transform("size")

    # Remove short trajectories
    df = df[df.traj_len > 150].copy()

    # Rename position column names. Input the column names of the object identifier and the position coordinates.
    column_names = {
        "object_identifier": "tracked_object",
        "x_position": "y_pos",
        "y_position": "x_pos",
    }

    inv_column_mapping = {
        "Pid": column_names["object_identifier"],
        "xf": column_names["x_position"],
        "yf": column_names["y_position"],
    }
    column_mapping = {v: k for k, v in inv_column_mapping.items()}
    df.rename(columns=column_mapping, inplace=True)

    # Convert position units to meters
    df["xf"] /= 1000
    df["yf"] /= 1000

    # df = add_velocity(df, groupby="Pid", xpos="xf", ypos="yf")
    return df


# def read_ehv_station_paths_from_azure(datehour: datetime.datetime, freq: str) -> pd.DataFrame:
#     import crowdflow as cf
#     from crowdflow.preprocessing.aggregators import KinematicsAggregator
#     from crowdflow.preprocessing.pipelines import siemens_trajectory_cleaner as cleaner

#     filename = "ehv_Perron2.1_siemens.json"
#     log.info("Reading %s", datehour)
#     area = cf.get_area(filename, **{"validate": False})
#     days = 1
#     if freq[1] == "d":
#         starttime = datetime.time(0)
#         endtime = datetime.time(23, 59)
#         days = int(freq[0])
#     elif freq[1] == "h":
#         starttime = datetime.time(datehour.hour)
#         endtime = datetime.time(datehour.hour + (int(freq[0]) - 1), 59)
#     df = pl.concat(
#         cf.read(
#             area,
#             "trajectorie",
#             "Siemens_Scan",
#             "azurescanpolars",
#             datehour.date(),
#             datehour.date() + datetime.timedelta(days=days),
#             starttime,
#             endtime,
#             **{"errors": "ignore", "show_progress": True},
#         )
#     )
#     df = cleaner(df, **{"NormalizeCoordinates.area": area})
#     df = df.collect()
#     ka = KinematicsAggregator(derivative="velocity")
#     df = ka.aggregate(df)
#     df = df.with_columns(pl.col("x_pos") / 1000)
#     df = df.with_columns(pl.col("y_pos") / 1000)
#     # df = df[df.traj_len > 150].copy()
#     return df.to_pandas()


def read_trajectories_from_path(filepath: Path) -> pd.DataFrame:
    """Read trajectories from file."""
    return pd.read_csv(filepath)


trajectory_reader = {
    "single_paths": read_single_paths,
    "parallel_paths": read_parallel_paths,
    "intersecting_paths": read_intersecting_paths,
    "intersecting_paths_synthetic": read_intersecting_paths_synthetic,
    "curved_paths": read_curved_paths,
    "curved_paths_synthetic": read_curved_paths_synthetic,
    "station_paths": read_station_paths,
    "asdz_pf34": read_asdz_pf34,
    # "minimal_dataset_for_testing": read_minimal_dataset_for_testing,
    # "ehv_azure": read_ehv_station_paths_from_azure,
}
