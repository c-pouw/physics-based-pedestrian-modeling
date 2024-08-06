"""Module to define utility functions for the configuration."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent


def apply_periodic_conditions_to_the_angle_theta(theta: float):
    """
    Apply periodic conditions to the angle theta.

    Args:
        theta (float): The angle theta.

    Returns:
        float: The angle theta after applying the periodic conditions.
    """
    theta += np.pi
    return theta % (2 * np.pi) - np.pi


def create_grid_name(grid_list: list):
    grid_list = [f"-{int(i*10)}" for i in grid_list]
    grid_name = "".join(grid_list)
    return grid_name


def set_plot_style(config: DictConfig, use_latex: bool = False) -> None:
    """Function to set the plot style.

    Args:
        use_latex: Whether to use LaTeX for the plot style or not. Defaults to False.
    """
    get_style = {True: "science", False: "science_no_latex"}
    style = get_style[use_latex]
    plt.style.use(Path(config.root_dir) / f"conf/{style}.mplstyle")


def register_new_resolvers(replace=False):
    OmegaConf.register_new_resolver("get_root_dir", lambda: ROOT_DIR, replace=replace)
    OmegaConf.register_new_resolver("parse_pi", lambda a: a * np.pi, replace=replace)
    OmegaConf.register_new_resolver(
        "generate_linear_bins", lambda min, max, step: np.arange(min, max + 0.01, step), replace=replace
    )
    OmegaConf.register_new_resolver(
        "generate_angular_bins", lambda min, segments: np.linspace(min, min + 2 * np.pi, segments + 1), replace=replace
    )
    OmegaConf.register_new_resolver("cast_numpy_array", np.array, replace=replace)
    OmegaConf.register_new_resolver(
        "apply_periodic_conditions_to_the_angle_theta", apply_periodic_conditions_to_the_angle_theta, replace=replace
    )
    OmegaConf.register_new_resolver("inv_prop", lambda x: 1 / x, replace=replace)
    OmegaConf.register_new_resolver("create_grid_name", create_grid_name, replace=replace)
    OmegaConf.register_new_resolver("set_plot_style", set_plot_style, replace=replace)


def initialize_hydra_config(env_name: str) -> DictConfig:
    """Function to initialize the Hydra configuration.

    Args:
        env_name: The name of the environment.
            For example: 'narrow_corridor', 'intersecting_paths', 'asdz_pf12', 'asdz_pf34', 'utrecht_pf5'.

    Returns:
        The Hydra configuration.
    """
    with initialize(version_base=None, config_path="../conf"):
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=[
                f"params={env_name}",
            ],
        )
        register_new_resolvers(replace=True)
    return config
