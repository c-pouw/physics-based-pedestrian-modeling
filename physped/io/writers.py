import logging
import pickle
from pathlib import Path

import pandas as pd

from physped.core.discrete_grid import PiecewisePotential
from physped.utils.functions import ensure_folder_exists

log = logging.getLogger(__name__)


def save_piecewise_potential(
    grid: PiecewisePotential, folderpath: Path, filename: str = "piecewise_potential.pickle"
) -> None:
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
