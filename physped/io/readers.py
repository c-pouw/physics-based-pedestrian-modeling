"""Data readers for the physics based pedestrian modeling code."""

import glob
import io
import logging
import pickle
import zipfile
from fnmatch import fnmatch
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from physped.core.piecewise_potential import PiecewisePotential

log = logging.getLogger(__name__)


def read_preprocessed_trajectories_from_file(config: DictConfig, **kwargs) -> pd.DataFrame:
    """Read preprocessed trajectories from a csv file.

    Mainly used to read intermediate outputs.

    Args:
        config: The configuration parameters.

    Returns:
        The preprocessed trajectory dataset.
    """
    filepath = Path.cwd().parent / config.filename.preprocessed_trajectories
    # if config.read.preprocessed_trajectories:
    #     log.debug("Configuration 'read.preprocessed_trajectories' is set to True.")
    try:
        preprocessed_trajectories = pd.read_csv(filepath)
        log.warning("Preprocessed trajectories read from file.")
        # log.debug("Filepath %s", filepath.relative_to(config.root_dir))
        return preprocessed_trajectories
    except FileNotFoundError as e:
        log.error("Preprocessed trajectories not found: %s", e)


def read_piecewise_potential(config: DictConfig) -> PiecewisePotential:
    filepath = Path.cwd().parent / config.filename.piecewise_potential
    if config.read.piecewise_potential:
        log.debug("Configuration 'read.simulated_trajectories' is set to True.")
        try:
            piecewise_potential = read_piecewise_potential_from_file(filepath)
            log.warning("Piecewise potential read from file")
            # log.debug("Filepath %s", filepath.relative_to(config.root_dir))
            return piecewise_potential
        except FileNotFoundError as e:
            log.error("Piecewise potential not found: %s", e)


def read_piecewise_potential_from_file(filepath: Path) -> PiecewisePotential:
    """Read piecewise potential from file.

    Args:
        filepath: Path to the file containing the piecewise potential.

    Returns:
        The piecewise potential.
    """
    with open(filepath, "rb") as file:
        piecewise_potential = pickle.load(file)
    return piecewise_potential


def read_narrow_corridor_paths_local(config: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read the narrow corridor paths archive from a local zip.

    The archive contains two files:

    - left-to-right.ssv: paths of pedestrians walking from left to right.
    - right-to-left.ssv: paths of pedestrians walking from right to left.


    Args:
        config: The configuration parameters.

    Returns:
        A tuple containing two DataFrames:
        - df_ltr: DataFrame for paths of pedestrians walking from left to right.
        - df_rtl: DataFrame for paths of pedestrians walking from right to left.
    """
    trajectory_data_dir = Path(config.trajectory_data_dir)
    log.info("Start reading single paths data set.")
    archive = zipfile.ZipFile(trajectory_data_dir / "data.zip")

    with archive.open("left-to-right.ssv") as paths_ltr:
        paths_ltr = paths_ltr.read().decode("utf-8")
    df_ltr = pd.read_csv(io.StringIO(paths_ltr), sep=" ")

    with archive.open("right-to-left.ssv") as paths_rtl:
        paths_rtl = paths_rtl.read().decode("utf-8")
    df_rtl = pd.read_csv(io.StringIO(paths_rtl), sep=" ")
    return df_ltr, df_rtl


def read_narrow_corridor_paths_4tu(config: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read the narrow corridor paths archive from 4TU remote repository.

    The archive contains two files:

    - left-to-right.ssv: paths of pedestrians walking from left to right.
    - right-to-left.ssv: paths of pedestrians walking from right to left.

    Args:
        config: The configuration parameters.

    Returns:
        A tuple containing two DataFrames:
        - df_ltr: DataFrame for paths of pedestrians walking from left to right.
        - df_rtl: DataFrame for paths of pedestrians walking from right to left.
    """
    link = "https://data.4tu.nl/ndownloader/items/b8e30f8c-3931-4604-842a-77c7fb8ac3fc/versions/1"
    bytestring = requests.get(link, timeout=10)
    with zipfile.ZipFile(io.BytesIO(bytestring.content), "r") as outerzip:
        with zipfile.ZipFile(outerzip.open("data.zip")) as innerzip:
            with innerzip.open("left-to-right.ssv") as paths_ltr:
                paths_ltr = paths_ltr.read().decode("utf-8")
            with innerzip.open("right-to-left.ssv") as paths_rtl:
                paths_rtl = paths_rtl.read().decode("utf-8")
    df_ltr = pd.read_csv(io.StringIO(paths_ltr), sep=" ")
    df_rtl = pd.read_csv(io.StringIO(paths_rtl), sep=" ")
    return df_ltr, df_rtl


narrow_corridor_path_reader = {
    "local": read_narrow_corridor_paths_local,
    "4tu": read_narrow_corridor_paths_4tu,
}


def read_intersecting_paths(config: DictConfig) -> pd.DataFrame:
    """Read the intersecting paths data set.

    The intersecting paths dataset is created by combining the left-to-right and right-to-left
    paths from the narrow corridor dataset. The paths from the right-to-left dataset are rotated
    by 90 degrees to create intersecting paths.

    Args:
        config: The configuration parameters.

    Returns:
        The trajectory dataset with intersecting paths.
    """
    data_source = config.params.data_source
    df_ltr, df_rtl = narrow_corridor_path_reader[data_source](config)

    df_ltr["X_SG"] = df_ltr["X_SG"] + 0.1
    df_ltr["Y_SG"] = df_ltr["Y_SG"] - 0.05

    df_rtl["X_SG"] = df_rtl["X_SG"] + 0.1
    df_rtl["Y_SG"] = df_rtl["Y_SG"] - 0.05

    # swap x and y coordinates to rotate by 90 degrees
    df_rtl.rename(columns={"X": "Y", "Y": "X", "X_SG": "Y_SG", "Y_SG": "X_SG", "U_SG": "V_SG", "V_SG": "U_SG"}, inplace=True)

    df = pd.concat([df_ltr, df_rtl], ignore_index=True)

    log.info("Finished reading single paths data set.")
    return df


def read_narrow_corridor_paths(config: DictConfig) -> pd.DataFrame:
    """Read the narrow corridor data set.

    The trajectories are read from local or remote sources based on the configuration.

    Args:
        config: The configuration parameters.

    Returns:
        The trajectory dataset with single paths.
    """
    data_source = config.params.data_source
    df_ltr, df_rtl = narrow_corridor_path_reader[data_source](config)

    df = pd.concat([df_ltr, df_rtl], ignore_index=True)

    df["X_SG"] = df["X_SG"] + 0.1
    df["Y_SG"] = df["Y_SG"] - 0.05

    # Only keep the columns that are needed
    df = df[["Pid", "Rstep", "X_SG", "Y_SG"]]
    log.info("Finished reading single paths data set.")
    return df


def read_wide_corridor_paths(config: DictConfig) -> pd.DataFrame:
    """Read the wide corridor data set from a local file.

    Args:
        config: The configuration parameters.

    Returns:
        The trajectory dataset with paths in the wide corridor.
    """
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
    return df


def read_curved_paths_synthetic(config: DictConfig) -> pd.DataFrame:
    """Read the synthetic curved paths data set.

    Args:
        config: The configuration parameters.

    Returns:
        The synthetic curved paths dataset.
    """
    root_dir = Path(config.root_dir).parent
    trajectory_data_dir = Path(config.trajectory_data_dir)
    file_path = root_dir / trajectory_data_dir / "artificial_measurements_ellipse.parquet"
    df = pd.read_parquet(file_path)
    df = df.rename(columns={"x": "xf", "y": "yf", "xdot": "uf", "ydot": "vf"})
    return df


# def filter_trajectory(df, cutoff=0.16, order=4):
#     b, a = signal.butter(order, cutoff, "low")
#     df = df.sort_values(["particle", "time"])
#     df = df.groupby("particle").filter(lambda x: len(x) > 52)

#     f_df = df.groupby(df["particle"]).apply(lambda x: pd.DataFrame(signal.filtfilt(b, a, x[["x", "y"]].values, axis=0)))
#     df[["x", "y"]] = f_df.set_index(df.index)
#     return df


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


def read_ehv_pf34_paths_geert(config: DictConfig) -> pd.DataFrame:
    """Read the Eindhoven platform 3-4 paths data set from Geert.

    Args:
        config: The configuration parameters.

    Returns:
        The trajectory dataset with Eindhoven platform 3-4 paths
    """
    root_dir = Path(config.root_dir).parent
    trajectory_data_dir = Path(config.trajectory_data_dir)
    file_path = root_dir / trajectory_data_dir / "trajectories_EHV_platform_2_1_refined.parquet"
    df = pd.read_parquet(file_path)
    df = df[["date_time_utc", "Pid", "xf", "yf"]]

    # Rotate the domain
    df.rename({"xf": "yf", "yf": "xf", "uf": "vf", "vf": "uf"}, axis=1, inplace=True)
    return df


def filter_part_of_the_domain(df, xmin, xmax, xcol="x_pos", ycol="y_pos"):
    df = df[df[xcol] > xmin].copy()
    df = df[df[ycol] < xmax].copy()
    return df


def read_ehv_pf34_paths_local(config: DictConfig) -> pd.DataFrame:
    """Read the Eindhoven platform 3-4 paths data set from a local file.

    Args:
        config: The configuration parameters.

    Returns:
        The trajectory dataset with Eindhoven platform 3-4 paths
    """
    trajectory_data_dir = Path(config.trajectory_data_dir)
    glob_string = f"{str(trajectory_data_dir)}/ehv_pf34/*.parquet"
    filelist = glob.glob(glob_string)
    df_list = []
    for file in tqdm(filelist, ascii=True):
        df = pd.read_parquet(file)
        df_list.append(df)

    df = pd.concat(df_list)

    # Rotate the domain by 90 degrees
    df.rename({"x_pos": "y_pos", "y_pos": "x_pos"}, axis=1, inplace=True)

    # Convert spatial coordinates from milimeters to meters
    df["x_pos"] /= 1000
    df["y_pos"] /= 1000

    df = filter_part_of_the_domain(df, xmin=50, xmax=70)
    return df


def get_zenodo_data(url: str) -> BytesIO:
    """Download data from Zenodo URL into memory"""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return BytesIO(response.content)


def list_zenodo_files(config: DictConfig, pattern: str = None) -> List[dict]:
    """List all files in a Zenodo record, optionally filtered by pattern

    Args:
        config: The configuration parameters.

    Returns:
        List of file information dictionaries
    """
    zenodo_repo_url = config.params.data_url
    data_record_id = config.params.data_record_id
    url = f"{zenodo_repo_url}/{data_record_id}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    files = response.json()["files"]

    if pattern:
        files = [f for f in files if fnmatch(f["key"], pattern)]

    return files


def read_parquet_from_zenodo(config: DictConfig) -> pd.DataFrame:
    """Read parquet file from Zenodo using Pandas

    Args:
        config: The configuration parameters.

    Returns:
        DataFrame from Pandas
    """
    # Get file information
    files = list_zenodo_files(config, "*.parquet")
    if not files:
        raise ValueError(f"No matching files found in record {config.params.record_id}")

    # Use first matching file if filename not specified
    file_info = files[0]  # TODO Reader for other files in the zenodo repo
    file_url = file_info["links"]["self"]

    # Get data into memory
    data = get_zenodo_data(file_url)

    # Read with specified library
    return pd.read_parquet(data, engine="pyarrow")


def read_ehv_pf34_paths_zenodo(config: DictConfig) -> pd.DataFrame:
    """Read the Eindhoven platform 3-4 paths data set from a local file.

    Args:
        config: The configuration parameters.

    Returns:
        The trajectory dataset with Eindhoven platform 3-4 paths
    """
    df = read_parquet_from_zenodo(config)
    df = df.reset_index(drop=True).head(10000000)  # TODO: Fix memory issues in a better way

    # Convert spatial coordinates from milimeters to meters
    xcol = config.params.colnames.xf
    ycol = config.params.colnames.yf
    df[xcol] /= 1000
    df[ycol] /= 1000

    df = filter_part_of_the_domain(df, xmin=50, xmax=70, xcol=xcol, ycol=ycol)
    return df


ehv_pf34_path_reader = {
    "geert": read_ehv_pf34_paths_geert,
    "local": read_ehv_pf34_paths_local,
    "zenodo": read_ehv_pf34_paths_zenodo,
}


def read_eindhoven_pf34_paths(config: DictConfig) -> pd.DataFrame:
    """Read the Eindhoven platform 3-4 paths data set.

    Args:
        config: The configuration parameters.

    Returns:
        The trajectory dataset with Eindhoven platform 3-4 paths.
    """
    path_reader = ehv_pf34_path_reader[config.params.data_source]
    df = path_reader(config)
    return df


def read_asdz_pf34_paths_local(config: DictConfig) -> pd.DataFrame:
    trajectory_data_dir = Path(config.trajectory_data_dir)
    file_path = trajectory_data_dir / "Amsterdam Zuid - platform 3-4 - set1.csv"
    df = pd.read_csv(file_path)
    return df


def read_asdz_pf34_paths_4tu(config: DictConfig) -> pd.DataFrame:
    link = "https://data.4tu.nl/file/7d78a5e3-6142-49fe-be03-e4c707322863/40ea5cd9-95dc-4e3c-8760-7f4dd543eae7"
    bytestring = requests.get(link, timeout=10)

    with zipfile.ZipFile(io.BytesIO(bytestring.content), "r") as zipped_file:
        with zipped_file.open("Amsterdam Zuid - platform 3-4 - set1.csv") as paths:
            df = pd.read_csv(paths)
    return df


asdz_pf34_path_reader = {
    "local": read_asdz_pf34_paths_local,
    "4tu": read_asdz_pf34_paths_4tu,
}


def read_asdz_pf34_paths(config: DictConfig) -> pd.DataFrame:
    df = asdz_pf34_path_reader[config.params.data_source](config)
    # Convert spatial coordinates from milimeters to meters
    df["x_pos"] /= 1000
    df["y_pos"] /= 1000
    return df


def read_utrecht_pf5_paths_4tu(config: DictConfig):
    """Read the Utrecht Centraal platform 5 paths data set from 4TU.

    Args:
        config: The configuration parameters.

    Returns:
        The trajectory dataset with Utrecht Centraal platform 5 paths
    """
    link = "https://data.4tu.nl/file/d4d548c6-d198-49b3-986c-e22319970a5e/a58041fb-0318-4bee-9b2c-934bd8e5df83"
    bytestring = requests.get(link, timeout=10)

    with zipfile.ZipFile(io.BytesIO(bytestring.content), "r") as zipped_file:
        with zipped_file.open("Utrecht Centraal - platform 5 - set99.csv") as paths:
            df = pd.read_csv(paths)
    return df


def read_utrecht_pf5_paths_local(config: DictConfig):
    """Read the Utrecht Centraal platform 5 paths data set from a local file.

    Args:
        config: The configuration parameters.

    Returns:
        The trajectory dataset with Utrecht Centraal platform 5 paths.
    """
    file_list = glob.glob(config.trajectory_data_dir + "/Utrecht*.csv")
    file_path = file_list[0]
    return pd.read_csv(file_path)


utrecht_pf5_path_reader = {
    "local": read_utrecht_pf5_paths_local,
    "4tu": read_utrecht_pf5_paths_4tu,
}


def read_utrecht_pf5_paths(config: DictConfig) -> pd.DataFrame:
    """Read the Utrecht Centraal platform 5 paths data set.

    The trajectories are read from local or remote sources based on the configuration.
    The spatial coordinates of the trajectories are converted from milimeters to meters.

    Args:
        config: The configuration parameters.

    Returns:
        The trajectory dataset with Utrecht Centraal platform 5 paths.
    """
    path_reader = utrecht_pf5_path_reader[config.params.data_source]
    df = path_reader(config)

    # Only keep the columns that are needed
    df = df[["x_pos", "y_pos", "tracked_object", "timestampms"]]

    # Convert spatial coordinates from milimeters to meters
    df["x_pos"] /= 1000
    df["y_pos"] /= 1000
    return df


def read_asdz_pf12_paths_4tu(config: DictConfig) -> pd.DataFrame:
    """Read the Amsterdam Zuid platform 1-2 paths data set from 4TU.

    Args:
        config: The configuration parameters.

    Returns:
        The trajectory dataset with Amsterdam Zuid platform 1-2 paths
    """
    link = "https://data.4tu.nl/file/af4ef093-69ef-4e1c-8fbc-c40c447c618c/ca88bfc5-5a79-496a-8c90-433fa40929b9"
    bytestring = requests.get(link, timeout=10)

    with zipfile.ZipFile(io.BytesIO(bytestring.content), "r") as zipped_file:
        with zipped_file.open("Amsterdam Zuid - platform 1-2 - set10.csv") as paths:
            df = pd.read_csv(paths)

    return df


def read_asdz_pf12_paths_local(config: DictConfig) -> pd.DataFrame:
    """Read the Amsterdam Zuid platform 1-2 paths data set from a local file.

    Args:
        config: The configuration parameters.

    Returns:
        The trajectory dataset with Amsterdam Zuid platform 1-2 paths
    """
    file_list = glob.glob(config.trajectory_data_dir + "/Amsterdam*Zuid*1-2*.csv")
    file_path = file_list[0]
    return pd.read_csv(file_path)


asdz_pf12_path_reader = {
    "local": read_asdz_pf12_paths_local,
    "4tu": read_asdz_pf12_paths_4tu,
}


def read_asdz_pf12_paths(config: DictConfig) -> pd.DataFrame:
    """Read the Amsterdam Zuid platform 1-2 paths data set.

    The trajectories are read from local or remote sources based on the configuration.
    The spatial coordinates of the trajectories are converted from milimeters to meters.

    Args:
        config: The configuration parameters.

    Returns:
        The trajectory dataset with Amsterdam Zuid platform 1-2 paths.
    """
    df = asdz_pf12_path_reader[config.params.data_source](config)

    # Convert spatial coordinates from milimeters to meters
    df["x_pos"] /= 1000
    df["y_pos"] /= 1000
    return df


trajectory_reader = {
    "narrow_corridor": read_narrow_corridor_paths,
    "wide_corridor": read_wide_corridor_paths,
    "intersecting_paths": read_intersecting_paths,
    "curved_paths_synthetic": read_curved_paths_synthetic,
    "eindhoven_pf34": read_eindhoven_pf34_paths,
    "asdz_pf34": read_asdz_pf34_paths,
    "utrecht_pf5": read_utrecht_pf5_paths,
    "asdz_pf12": read_asdz_pf12_paths,
}


def get_background_image_local(config: DictConfig) -> Image.Image:
    """Read the background image from a local file.

    Args:
        config: The configuration parameters.

    Returns:
        The background image as a PIL Image object.
    """
    image = plt.imread(Path(config.root_dir).parent / config.params.background.imgpath)
    return image


def get_background_image_from_remote_zip(config: DictConfig) -> Image.Image:
    """Read the background image from a remote archive.

    Args:
        config: The configuration parameters.

    Returns:
        The background image as a PIL Image object.
    """
    link = config.params.background.img_link_4tu
    bytestring = requests.get(link, timeout=10)

    archive = zipfile.ZipFile(io.BytesIO(bytestring.content), "r")
    background_name = [x for x in archive.namelist() if x.endswith(".png")][0]

    with archive.open(background_name) as contents:
        image_data = contents.read()

    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = plt.imread(io.BytesIO(image_array))
    return image


def get_background_image_from_zenodo(config: DictConfig, filename: str = None) -> Image.Image:
    """Read image file from Zenodo using PIL

    Args:
        config: The configuration parameters.
        filename: Name of file to read. If None, will read first image file found

    Returns:
        PIL Image object
    """
    # Get file information
    pattern = filename if filename else "*.png"  # Can be extended to support other formats
    files = list_zenodo_files(config, pattern)
    if not files:
        raise ValueError(f"No matching image files found in record {config.params.data_record_id}")

    # Use first matching file if filename not specified
    file_info = files[0]
    file_url = file_info["links"]["self"]

    # Get data into memory
    data = get_zenodo_data(file_url)

    # Read image using PIL
    return Image.open(data)


read_background_image = {
    "local": get_background_image_local,
    "4tu": get_background_image_from_remote_zip,
    "zenodo": get_background_image_from_zenodo,
}
