import logging
from pathlib import Path

import numpy as np
import pandas as pd

from physped.core.functions_to_discretize_grid import convert_grid_indices_to_coordinates, sample_from_ndarray
from physped.core.langevin_model import LangevinModel
from physped.io.writers import save_trajectories
from physped.utils.functions import cart2pol

log = logging.getLogger(__name__)


def sample_trajectory_origins_from_heatmap(piecewise_potential, parameters: dict) -> np.ndarray:
    origins = sample_from_ndarray(piecewise_potential.histogram[..., 0], parameters.simulation.ntrajs)
    origins = np.hstack((origins, np.zeros((origins.shape[0], 1), dtype=int)))
    origins = convert_grid_indices_to_coordinates(piecewise_potential, origins)
    origins = np.hstack((origins, origins))
    origins = np.delete(origins, 4, axis=1)
    return origins


def simulate_trajectories(piecewise_potential, parameters: dict) -> pd.DataFrame:
    origins = sample_trajectory_origins_from_heatmap(piecewise_potential, parameters)
    # Simulate trajectories
    lm = LangevinModel(piecewise_potential, parameters)
    t_eval = np.arange(parameters.simulation.start, parameters.simulation.end, parameters.simulation.step)
    # simulation_time = params.get("simulation_time", [0, 10, 0.1])
    # t_eval = np.arange(*simulation_time)
    trajectories = []
    for Pid, X_0 in enumerate(origins[:, :8]):
        solution = lm.simulate(X_0, t_eval)
        traj = pd.DataFrame(solution, columns=["xf", "yf", "uf", "vf", "xs", "ys", "us", "vs"]).dropna()
        traj["Pid"] = Pid
        traj["t"] = t_eval[: len(traj)]
        traj["k"] = range(len(traj))
        trajectories.append(traj)

    trajectories = pd.concat(trajectories)
    trajectories["rf"], trajectories["thetaf"] = cart2pol(trajectories.uf, trajectories.vf)
    trajectories["rs"], trajectories["thetas"] = cart2pol(trajectories.us, trajectories.vs)
    folderpath = Path(parameters.folder_path)
    save_trajectories(trajectories, folderpath, "simulated_trajectories.csv")
    return trajectories