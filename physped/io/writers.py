import pickle
import json
from pathlib import Path

import logging

from physped.core.discrete_grid import DiscreteGrid


log = logging.getLogger(__name__)


def save_discrete_grid(grid: DiscreteGrid, folderpath: Path):
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
