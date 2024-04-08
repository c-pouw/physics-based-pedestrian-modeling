import numpy as np
from omegaconf import OmegaConf


def register_new_resolvers():
    OmegaConf.register_new_resolver("parse_pi", lambda a: a * np.pi)
    OmegaConf.register_new_resolver(
        "generate_linear_bins", lambda min, max, step: np.arange(min, max + 0.01, step)
    )
    OmegaConf.register_new_resolver(
        "generate_angular_bins", lambda min, segments: np.linspace(min, min + 2 * np.pi + 0.01, segments + 1)
    )
    OmegaConf.register_new_resolver("cast_numpy_array", np.array)
