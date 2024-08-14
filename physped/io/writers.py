import logging
import pickle
from pathlib import Path

import pandas as pd

from physped.core.piecewise_potential import PiecewisePotential

log = logging.getLogger(__name__)


def save_piecewise_potential(grid: PiecewisePotential, folderpath: Path, filename: str = "piecewise_potential.pickle") -> None:
    """Save piecewise potential

    Args:
        grid: The piecewise potential to save.
        folderpath: The folder to save the piecewise potential in.
        filename: The filenam to save the piecewise potential in.
        Defaults to "piecewise_potential.pickle".
    """
    filepath = folderpath / filename
    with open(filepath, "wb") as f:
        pickle.dump(grid, f)
    log.info("Piecewise potential saved as %s.", filename)


def save_trajectories(trajectories: pd.DataFrame, folderpath: Path, filename: str) -> None:
    """Save trajectories

    Args:
        trajectories: The trajectories to save.
        folderpath: The folder to save the trajectories in.
        filename: The name of the file to save the trajectories in.
    """
    filepath = folderpath / filename
    trajectories.to_csv(filepath)
    log.info("Trajectories saved as %s.", filename)
