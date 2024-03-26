import json
import logging
import pickle
from pathlib import Path

import pandas as pd

from physped.core.discrete_grid import PiecewisePotential
from physped.utils.functions import ensure_folder_exists

log = logging.getLogger(__name__)


def save_piecewise_potential(grid: PiecewisePotential, folderpath: Path, filename: str = "model.pickle") -> None:
    """
    Save a PiecewisePotential object to a file using pickle.

    :param grid: The PiecewisePotential object to save.
    :type grid: PiecewisePotential
    :param folderpath: The path to the folder to save the file in.
    :type folderpath: Path

    :return: None
    """
    filepath = folderpath / filename
    with open(filepath, "wb") as f:
        pickle.dump(grid, f)
    log.info("Validation model saved to %s.", filepath.relative_to(Path.cwd()))


def _save_parameters(parameters: dict, folderpath: Path) -> None:
    # ! Hydra made this obsolete
    """
    Save a dictionary of parameters to a JSON file.

    Parameters:
    - parameters (dict): The dictionary of parameters to save.
    - folderpath (str): The path to the folder to save the file in.

    Returns:
    - None
    """
    filepath = folderpath / "parameters.json"
    with open(filepath, "w") as f:
        json.dump(parameters, f, indent=4)
    log.info("Parameters saved to %s.", filepath.relative_to(Path.cwd()))


def save_trajectories(trajectories: pd.DataFrame, folderpath: Path, filename: str) -> None:
    """
    Save trajectories to a CSV file.

    :param trajectories: The DataFrame containing the trajectories to save.
    :type trajectories: pd.DataFrame
    :param folderpath: The path to the folder to save the file in.
    :type folderpath: Path

    :return: None
    """
    ensure_folder_exists(folderpath)
    filepath = folderpath / filename
    trajectories.to_csv(filepath)
    log.info("Trajectories saved to %s.", filepath.relative_to(Path.cwd()))


# def save_preprocessed_trajectories(trajectories: pd.DataFrame, folderpath: Path) -> None:
#     # TODO: Combine with save_simulated_trajectories
#     """Save preprocessed trajectories to a CSV file.

#     This function takes a DataFrame containing preprocessed trajectories and a folder path,
#     and saves the trajectories to a CSV file in the specified folder.

#     :param trajectories: The DataFrame containing the preprocessed trajectories.
#     :type trajectories: pd.DataFrame
#     :param folderpath: The path to the folder where the CSV file will be saved.
#     :type folderpath: Path
#     """
#     filepath = folderpath / "preprocessed_trajectories.csv"
#     trajectories.to_csv(filepath)
#     log.info(
#         "Saved preprocessed trajectories to %s",
#         filepath.relative_to(Path.cwd()),
#     )


# def save_simulated_trajectories(trajectories: pd.DataFrame, folderpath: Path) -> None:
#     """Save simulated trajectories to a CSV file.

#     This function takes a DataFrame containing simulated trajectories and a folder path,
#     and saves the trajectories to a CSV file in the specified folder.

#     :param trajectories: The DataFrame containing the simulated trajectories.
#     :type trajectories: pd.DataFrame
#     :param folderpath: The path to the folder where the CSV file will be saved.
#     :type folderpath: Path
#     """
#     filepath = folderpath / "simulated_trajectories.csv"
#     trajectories.to_csv(filepath)
#     log.info(
#         "Saved %d trajectories to %s",
#         len(trajectories.Pid.unique()),
#         filepath.relative_to(Path.cwd()),
#     )
