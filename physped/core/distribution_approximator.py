"""Module for a with distribution approximation classes.

"""

import logging
from dataclasses import dataclass
from typing import Callable, Tuple

from scipy.stats import norm

log = logging.getLogger(__name__)


@dataclass
class DistApproximation:
    """A class for the distribution approximation of the potential."""

    fit_dimensions: Tuple[str, ...]
    fit_parameters: Tuple[str, ...]
    function: Callable


class GaussianApproximation(DistApproximation):
    def __init__(self):
        predefined_kwargs = {"fit_dimensions": ("x", "y", "u", "v"), "fit_parameters": ("mu", "sigma"), "function": norm.fit}
        super().__init__(**predefined_kwargs)
