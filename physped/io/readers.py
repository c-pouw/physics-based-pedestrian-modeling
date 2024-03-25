"""Trajectory readers for the pathintegral code."""

import logging
import pickle
import zipfile
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from physped.core.discrete_grid import DiscretePotential
from physped.utils.functions import add_velocity

trajectory_folder_path = Path.cwd() / "data" / "trajectories"

log = logging.getLogger(__name__)


def read_grid_bins(grid_name: str):
    filename = f"data/grids/{grid_name}.npz"
    return np.load(filename, allow_pickle=True)


def read_discrete_grid_from_file(filename: Path) -> DiscretePotential:
    """
    Read a validation model from a file using pickle.

    Parameters:
    - filename (str): The path to the file containing the validation model.

    Returns:
    - The validation model object.
    """
    with open(filename, "rb") as f:
        val = pickle.load(f)
    log.info("Successfully read `%s` validation model.", filename.relative_to(Path.cwd()))
    return val


def single_paths() -> pd.DataFrame:
    """Read the single paths data set."""
    # Specify the file path
    log.info("Start reading single paths data set.")
    # Open the zip file
    archive = zipfile.ZipFile(trajectory_folder_path / "data.zip")

    # Read the .ssv file as a string
    with archive.open("left-to-right.ssv") as f:
        data_str = f.read().decode("utf-8")

    # Convert the string to a pandas DataFrame
    df1 = pd.read_csv(StringIO(data_str), sep=" ")

    # Read the .ssv file as a string
    with archive.open("right-to-left.ssv") as f:
        data_str = f.read().decode("utf-8")

    # Convert the string to a pandas DataFrame
    df2 = pd.read_csv(StringIO(data_str), sep=" ")
    df = pd.concat([df1, df2], ignore_index=True)
    df.rename(columns={"X_SG": "xf", "Y_SG": "yf", "U_SG": "uf", "V_SG": "vf"}, inplace=True)
    log.info("Finished reading single paths data set.")
    return df


def parallel_paths() -> pd.DataFrame:
    """Read the parallel paths data set."""
    file_path = trajectory_folder_path / "df_single_pedestrians_small.h5"
    df = pd.read_hdf(file_path)
    df.rename(columns={"X_SG": "xf", "Y_SG": "yf", "U_SG": "uf", "V_SG": "vf"}, inplace=True)
    df["Pid"] = df.groupby(["Pid", "day_id"]).ngroup()
    df = df.query("Umean>0.5").loc[abs(df.X0 - df.X1) > 2]
    df = df.groupby("Pid").filter(lambda x: max(x.uf) < 3.5)
    return df


def intersecting_paths() -> pd.DataFrame:
    """Read the intersecting paths data set."""
    file_path = trajectory_folder_path / "simulations_crossing.parquet"
    df = pd.read_parquet(file_path)
    df.rename(columns={"X_SG": "xf", "Y_SG": "yf", "U_SG": "uf", "V_SG": "vf"}, inplace=True)
    df["k"] = df.groupby("Pid").cumcount()
    return df


def curved_paths() -> pd.DataFrame:
    """Read the curved paths data set."""
    file_path = trajectory_folder_path / "artificial_measurements_ellipse.parquet"
    df = pd.read_parquet(file_path)
    df = df.rename(columns={"x": "xf", "y": "yf", "xdot": "uf", "ydot": "vf"})
    return df


def station_paths() -> pd.DataFrame:
    """Read the station paths data set."""
    file_path = trajectory_folder_path / "trajectories_EHV_platform_2_1_refined.parquet"
    df = pd.read_parquet(file_path)
    df.rename({"xf": "yf", "yf": "xf", "uf": "vf", "vf": "uf"}, axis=1, inplace=True)
    return df


def ehv_train_station_multifile(filelist) -> pd.DataFrame:
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

    df = add_velocity(df, groupby="Pid", xpos="xf", ypos="yf")
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


def read_preprocessed_trajectories(folderpath: str) -> pd.DataFrame:
    """Read preprocessed trajectories from file."""
    filepath = Path(folderpath) / "preprocessed_trajectories.csv"
    trajectories = pd.read_csv(filepath)
    log.info(
        "Succesfully read preprocessed trajectories %s.",
        filepath.relative_to(Path.cwd()),
    )
    return trajectories


def read_simulated_trajectories(folderpath: str) -> pd.DataFrame:
    """Read simulated trajectories from file."""
    filepath = Path(folderpath) / "simulated_trajectories.csv"
    trajectories = pd.read_csv(filepath)
    log.info(
        "Succesfully read simulated trajectories %s.",
        filepath.relative_to(Path.cwd()),
    )
    return trajectories


trajectory_reader = {
    "single_paths": single_paths,
    "parallel_paths": parallel_paths,
    "intersecting_paths": intersecting_paths,
    "curved_paths": curved_paths,
    "station_paths": station_paths,
    # "ehv_azure": read_ehv_station_paths_from_azure,
}
