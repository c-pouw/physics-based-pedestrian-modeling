import json
import logging
import pickle
from pathlib import Path

import pandas as pd

from physped.core.discrete_grid import DiscretePotential

log = logging.getLogger(__name__)


def save_discrete_potential(grid: DiscretePotential, folderpath: Path):
    """
    Save a DiscreteGrid object to a file using pickle.

    Parameters:
    - grid (DiscreteGrid): The DiscreteGrid object to save.
    - filepath (str): The path to the file to save the object to.

    Returns:
    - None
    """
    filepath = folderpath / "model.pickle"
    with open(filepath, "wb") as f:
        pickle.dump(grid, f)
    log.info("Validation model saved to %s.", filepath.relative_to(Path.cwd()))


def save_parameters(parameters: dict, folderpath: Path) -> None:
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


def save_preprocessed_trajectories(trajectories: pd.DataFrame, folderpath: Path) -> None:
    """Save preprocessed trajectories to a CSV file.

    This function takes a DataFrame containing preprocessed trajectories and a folder path,
    and saves the trajectories to a CSV file in the specified folder.

    :param trajectories: The DataFrame containing the preprocessed trajectories.
    :type trajectories: pd.DataFrame
    :param folderpath: The path to the folder where the CSV file will be saved.
    :type folderpath: Path
    """
    filepath = folderpath / "preprocessed_trajectories.csv"
    trajectories.to_csv(filepath)
    log.info(
        "Saved preprocessed trajectories to %s",
        filepath.relative_to(Path.cwd()),
    )


def save_simulated_trajectories(trajectories: pd.DataFrame, folderpath: Path) -> None:
    """Save simulated trajectories to a CSV file.

    This function takes a DataFrame containing simulated trajectories and a folder path,
    and saves the trajectories to a CSV file in the specified folder.

    :param trajectories: The DataFrame containing the simulated trajectories.
    :type trajectories: pd.DataFrame
    :param folderpath: The path to the folder where the CSV file will be saved.
    :type folderpath: Path
    """
    filepath = folderpath / "simulated_trajectories.csv"
    trajectories.to_csv(filepath)
    log.info(
        "Saved %d trajectories to %s",
        len(trajectories.Pid.unique()),
        filepath.relative_to(Path.cwd()),
    )
