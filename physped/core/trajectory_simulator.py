import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from physped.core.langevin_model import LangevinModel
from physped.core.parametrize_potential import get_grid_indices
from physped.core.pedestrian_initializer import (
    sample_trajectory_origins_from_heatmap,
    sample_trajectory_origins_from_trajectory_state_n,
)
from physped.core.piecewise_potential import PiecewisePotential
from physped.io.writers import save_trajectories
from physped.utils.functions import cartesian_to_polar_coordinates, periodic_angular_conditions

log = logging.getLogger(__name__)


# def sample_trajectory_origins_from_trajectories(piecewise_potential: PiecewisePotential, parameters: dict) -> np.ndarray:
#     ntrajs = np.min([parameters.simulation.ntrajs, parameters.input_ntrajs])
#     sampled_origins = piecewise_potential.trajectory_origins.sample(n=ntrajs)
#     # Stacking to go from [x,y,u,v] to [x,y,u,v,xs,ys,us,vs]
#     sampled_origins = np.hstack((sampled_origins, sampled_origins))
#     log.info("Sampled %d origins from the input trajectories.", ntrajs)
#     return sampled_origins


def read_simulated_trajectories_from_file(config):
    log.debug("Configuration 'read.simulated_trajectories' is set to True.")
    filepath = Path.cwd().parent / config.filename.simulated_trajectories
    try:
        simulated_trajectories = pd.read_csv(filepath)
        log.warning("Simulated trajectories read from file")
        # log.debug("Filepath %s", filepath.relative_to(config.root_dir))
        return simulated_trajectories
    except FileNotFoundError as e:
        log.error("Preprocessed trajectories not found: %s", e)


def heatmap_zero_at_slow_state(piecewise_potential: PiecewisePotential, slow_state) -> bool:
    """Check if the heatmap is zero at the given position.

    Parameters:
        piecewise_potential: The piecewise potential object.
        position: The position to check.

    Returns:
        True if the heatmap is zero at the given position, False otherwise.
    """
    heatmap = np.sum(piecewise_potential.histogram, axis=(2, 3, 4))
    slow_state_index = get_grid_indices(piecewise_potential, slow_state)
    return heatmap[slow_state_index[0], slow_state_index[1]] == 0


def simulate_trajectories(piecewise_potential: PiecewisePotential, config: dict, measured_trajectories: None) -> pd.DataFrame:
    parameters = config.params

    if config.read.simulated_trajectories:
        return read_simulated_trajectories_from_file(config)

    log.info("Simulate trajectories using the piecewise potential")
    # TODO : Can we create a dictionary with these functions if they have different function arguments?
    match config.params.simulation.sample_origins_from:
        case "heatmap":
            log.warning("Trajectory origins will be sampled from a heatmap.")
            origins = sample_trajectory_origins_from_heatmap(piecewise_potential, parameters)
        case "trajectories":
            sample_state = config.params.simulation.sample_state
            log.warning("Simulation origins will be sampled from measured trajectories at state %s", sample_state)
            origins = sample_trajectory_origins_from_trajectory_state_n(
                parameters, measured_trajectories, sample_state, piecewise_potential
            )
    # Add t=0 to the end of the origins array
    start_time = np.zeros((origins.shape[0], 1))
    origins = np.hstack((origins, start_time))
    start_k = np.zeros((origins.shape[0], 1))
    origins = np.hstack((origins, start_k))
    Pids = np.arange(origins.shape[0])
    origins = np.hstack((origins, Pids[:, None]))

    evaluation_time = np.arange(parameters.simulation.start, parameters.simulation.end, parameters.simulation.step)
    trajectories = []

    model = LangevinModel(piecewise_potential, parameters)
    n_frames_back = config.params.fps // 2  # Go 0.5 second back
    with logging_redirect_tqdm():
        for starting_state in tqdm(
            origins[:, :11], desc="Simulating trajectories", unit="trajs", total=origins.shape[0], miniters=1
        ):
            first_trajectory_piece = simulate_trajectory_piece(model, starting_state, evaluation_time, 0)
            pid = int(first_trajectory_piece.iloc[0]["Pid"])
            trajectory_pieces = [first_trajectory_piece]

            while check_restarting_conditions(trajectory_pieces[-1], n_frames_back, piecewise_potential):
                last_trajectory_piece = trajectory_pieces[-1]
                no_last_trajectory_piece = int(last_trajectory_piece.iloc[0]["piece_id"])
                restarting_state = last_trajectory_piece.iloc[-n_frames_back]
                restarting_time = restarting_state["t"]
                log.info(
                    "Pid %s piece %s: Reverting %s frames -> t = %.2f.",
                    int(pid),
                    no_last_trajectory_piece,
                    n_frames_back,
                    restarting_time,
                )

                last_trajectory_piece = last_trajectory_piece.iloc[
                    : -n_frames_back - 1
                ]  # strip frames from last trajectory piece
                trajectory_pieces[-1] = last_trajectory_piece
                frame_to_restart_from = int(restarting_state["k"])

                new_evaluation_time = evaluation_time[frame_to_restart_from:]

                no_new_traj_piece = no_last_trajectory_piece + 1
                new_trajectory_piece = simulate_trajectory_piece(model, restarting_state, new_evaluation_time, no_new_traj_piece)
                trajectory_pieces.append(new_trajectory_piece)

            trajectory_pieces = pd.concat(trajectory_pieces)
            trajectories.append(trajectory_pieces)

    trajectories = pd.concat(trajectories).dropna()
    trajectories["rf"], trajectories["thetaf"] = cartesian_to_polar_coordinates(trajectories.uf, trajectories.vf)
    trajectories["rs"], trajectories["thetas"] = cartesian_to_polar_coordinates(trajectories.us, trajectories.vs)
    trajectories["thetaf"] = periodic_angular_conditions(trajectories["thetaf"], config.params.grid.bins["theta"])
    trajectories["thetas"] = periodic_angular_conditions(trajectories["thetas"], config.params.grid.bins["theta"])
    if config.save.simulated_trajectories:
        log.debug("Configuration 'save.simulated_trajectories' is set to True.")
        save_trajectories(trajectories, Path.cwd().parent, config.filename.simulated_trajectories)
    return trajectories


def check_restarting_conditions(traj, n_frames_back, piecewise_potential):
    # traj is the last trajectory_piece
    last_trajectory_piece_too_short = len(traj) < n_frames_back
    if last_trajectory_piece_too_short:
        log.warning("Last trajectory piece has only %s frames. Not reverting.", len(traj))
        return False

    trajectory_already_left_the_measurement_domain = heatmap_zero_at_slow_state(
        piecewise_potential, traj.iloc[-1][["xs", "ys", "us", "vs", "k"]]
    )
    if trajectory_already_left_the_measurement_domain:
        log.warning("Last trajectory piece already left the measurement domain. Not reverting.")
        return False

    particle_left_the_lattice = np.all(np.isinf(traj.iloc[-1]))
    if particle_left_the_lattice:
        log.warning("Last trajectory piece already left the lattice. Not reverting.")
        return False

    max_restarts = 2
    simulation_already_restarted_too_often = traj["piece_id"].iloc[0] > max_restarts
    if simulation_already_restarted_too_often:
        log.warning("Simulation already restarted %s times. Not reverting.", max_restarts)
        return False

    return True


def simulate_trajectory_piece(
    model: LangevinModel, starting_state: np.ndarray, evaluation_time: np.ndarray, no_traj_piece: int
) -> pd.DataFrame:
    integration_output = model.simulate(starting_state[:11], evaluation_time)
    columns = ["xf", "yf", "uf", "vf", "xs", "ys", "us", "vs", "t", "k", "Pid"]
    trajectory_piece = pd.DataFrame(integration_output, columns=columns)
    trajectory_piece.replace([np.inf, -np.inf], np.nan, inplace=True)
    trajectory_piece.dropna(inplace=True)
    trajectory_piece["piece_id"] = no_traj_piece
    return trajectory_piece
