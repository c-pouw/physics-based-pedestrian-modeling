import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from physped.core.functions_to_discretize_grid import convert_grid_indices_to_coordinates, sample_from_ndarray
from physped.core.langevin_model import LangevinModel
from physped.io.readers import read_trajectories_from_path
from physped.io.writers import save_trajectories
from physped.utils.functions import cart2pol

log = logging.getLogger(__name__)


def sample_trajectory_origins_from_heatmap(piecewise_potential, parameters: dict) -> np.ndarray:
    sampled_origins = sample_from_ndarray(piecewise_potential.histogram[..., 0], parameters.simulation.ntrajs)
    sampled_origins = np.hstack((sampled_origins, np.zeros((sampled_origins.shape[0], 1), dtype=int)))
    sampled_origins = convert_grid_indices_to_coordinates(piecewise_potential, sampled_origins)
    sampled_origins = np.hstack((sampled_origins, sampled_origins))
    sampled_origins = np.delete(sampled_origins, 4, axis=1)
    return sampled_origins


def sample_trajectory_origins_from_trajectories(piecewise_potential, parameters: dict) -> np.ndarray:
    ntrajs = np.min([parameters.simulation.ntrajs, parameters.input_ntrajs])
    sampled_origins = piecewise_potential.trajectory_origins.sample(n=ntrajs)
    # Stacking to go from [x,y,u,v] to [x,y,u,v,xs,ys,us,vs]
    sampled_origins = np.hstack((sampled_origins, sampled_origins))
    log.info("Sampled %d origins from the input trajectories.", ntrajs)
    return sampled_origins


def simulate_trajectories(piecewise_potential, config: dict) -> pd.DataFrame:
    parameters = config.params
    filepath = Path.cwd().parent / config.filename.simulated_trajectories

    # TODO : Move to separate function
    if config.read.simulated_trajectories:
        log.debug("Configuration 'read.simulated_trajectories' is set to True.")
        try:
            simulated_trajectories = read_trajectories_from_path(filepath)
            log.warning("Simulated trajectories read from file")
            log.debug("Filepath %s", filepath.relative_to(config.root_dir))
            return simulated_trajectories
        except FileNotFoundError as e:
            log.error("Preprocessed trajectories not found: %s", e)

    log.info("Simulate trajectories using the piecewise potential")
    match config.params.simulation.sample_origins_from:
        case "heatmap":
            log.warning("Trajectory origins will be sampled from a heatmap.")
            origins = sample_trajectory_origins_from_heatmap(piecewise_potential, parameters)
        case "trajectories":
            log.warning("Trajectory origins will be sampled from the input trajectories.")
            origins = sample_trajectory_origins_from_trajectories(piecewise_potential, parameters)

    lm = LangevinModel(piecewise_potential, parameters)
    t_eval = np.arange(parameters.simulation.start, parameters.simulation.end, parameters.simulation.step)
    trajectories = []
    Pid = 0
    with logging_redirect_tqdm():
        for X_0 in tqdm(origins[:, :8], desc="Simulating trajectories", unit="trajs", total=origins.shape[0], miniters=10):
            solution = lm.simulate(X_0, t_eval)
            traj = pd.DataFrame(solution, columns=["xf", "yf", "uf", "vf", "xs", "ys", "us", "vs"]).dropna()
            traj["Pid"] = Pid
            traj["t"] = t_eval[: len(traj)]
            traj["k"] = range(len(traj))
            trajectories.append(traj)
            Pid += 1

    trajectories = pd.concat(trajectories)
    trajectories["rf"], trajectories["thetaf"] = cart2pol(trajectories.uf, trajectories.vf)
    trajectories["rs"], trajectories["thetas"] = cart2pol(trajectories.us, trajectories.vs)
    if config.save.simulated_trajectories:
        log.debug("Configuration 'save.simulated_trajectories' is set to True.")
        save_trajectories(trajectories, Path.cwd().parent, config.filename.simulated_trajectories)
    return trajectories
