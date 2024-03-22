import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import hydra

import physped as pp
from physped.utils.functions import cart2pol

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf")
def simulate_trajectories(cfg):  # name: str, no_trajs: int):
    """Simulate trajectories and save them to a file.

    Args:
        name (str): The name of the environment to simulate trajectories for.
        no_trajs (int): The number of trajectories to simulate.
    """
    # Read grid and parameters
    # params = pp.read_parameter_file(name)
    folderpath = Path(cfg.params.folder_path)

    grids = pp.read_discrete_grid_from_file(folderpath / "model.pickle")

    # Create origins positions
    origins = pp.sample_from_ndarray(grids.histogram[..., 0], cfg.params.simulation.ntrajs)
    origins = np.hstack((origins, np.zeros((origins.shape[0], 1), dtype=int)))
    origins = pp.convert_grid_indices_to_coordinates(grids, origins)
    origins = np.hstack((origins, origins))
    origins = np.delete(origins, 4, axis=1)

    # Simulate trajectories
    lm = pp.LangevinModel(grids, cfg.params)
    t_eval = np.arange(cfg.params.simulation.start, cfg.params.simulation.end, cfg.params.simulation.step)
    # simulation_time = params.get("simulation_time", [0, 10, 0.1])
    # t_eval = np.arange(*simulation_time)
    trajs = []
    for Pid, X_0 in enumerate(origins[:, :8]):
        solution = lm.simulate(X_0, t_eval)
        traj = pd.DataFrame(solution, columns=["xf", "yf", "uf", "vf", "xs", "ys", "us", "vs"]).dropna()
        traj["Pid"] = Pid
        traj["t"] = t_eval[: len(traj)]
        traj["k"] = range(len(traj))
        trajs.append(traj)

    trajs = pd.concat(trajs)
    trajs["rf"], trajs["thetaf"] = cart2pol(trajs.uf, trajs.vf)
    trajs["rs"], trajs["thetas"] = cart2pol(trajs.us, trajs.vs)

    # Save trajectories
    filepath = Path(folderpath) / "simulated_trajectories.csv"
    trajs.to_csv(filepath)
    log.info(
        "Saved %d trajectories to %s",
        len(trajs.Pid.unique()),
        filepath.relative_to(Path.cwd()),
    )


if __name__ == "__main__":
    simulate_trajectories()
