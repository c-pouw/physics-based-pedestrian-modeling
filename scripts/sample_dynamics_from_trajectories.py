import argparse
import logging

import numpy as np

from physped.core.pedestrian_initializer import (
    sample_dynamics_from_trajectories,
)
from physped.core.slow_dynamics import compute_slow_dynamics
from physped.io.readers import trajectory_reader
from physped.preprocessing.trajectories import preprocess_trajectories
from physped.utils.config_utils import initialize_hydra_config

log = logging.getLogger(__name__)


def sample_and_save_dynamics_from_trajectories(env_name, n_trajs, state_n):
    config = initialize_hydra_config(env_name)
    trajectories = trajectory_reader[env_name](config)
    preprocessed_trajectories = preprocess_trajectories(
        trajectories, config=config
    )
    preprocessed_trajectories = compute_slow_dynamics(
        preprocessed_trajectories, config=config
    )

    dynamics = sample_dynamics_from_trajectories(
        config.params, preprocessed_trajectories, n_trajs, state_n
    )
    foldername = "data/intermediate/"
    filename = f"{env_name}_state_{state_n}_dynamics.npy"
    np.save(foldername + filename, dynamics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI script")
    parser.add_argument(
        "--env_name", type=str, help="Environment name", required=True
    )
    parser.add_argument(
        "--n_trajs",
        type=int,
        help="Number of trajectories",
        required=False,
        default=20,
    )
    parser.add_argument(
        "--state_n",
        type=int,
        help="the state to sample",
        required=False,
        default=0,
    )
    args = parser.parse_args()
    sample_and_save_dynamics_from_trajectories(
        args.env_name, args.n_trajs, args.state_n
    )
