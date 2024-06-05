import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent

# ! Todo this file move to utils


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
