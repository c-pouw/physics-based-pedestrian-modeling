import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

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


def register_new_resolvers():
    OmegaConf.register_new_resolver("get_root_dir", lambda: ROOT_DIR)
    OmegaConf.register_new_resolver("parse_pi", lambda a: a * np.pi)
    OmegaConf.register_new_resolver(
        "generate_linear_bins", lambda min, max, step: np.arange(min, max + 0.01, step)
    )
    OmegaConf.register_new_resolver(
        "generate_angular_bins", lambda min, segments: np.linspace(min, min + 2 * np.pi + 0.01, segments + 1)
    )
    OmegaConf.register_new_resolver("cast_numpy_array", np.array)
    OmegaConf.register_new_resolver(
        "apply_periodic_conditions_to_the_angle_theta", apply_periodic_conditions_to_the_angle_theta
    )
