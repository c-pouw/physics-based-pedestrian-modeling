"""Langevin model class."""

import logging

import numpy as np
import sdeint

from physped.core.functions_to_discretize_grid import get_grid_indices
from physped.core.piecewise_potential import PiecewisePotential
from physped.utils.functions import cart2pol, digitize_values_to_grid

log = logging.getLogger(__name__)


SLOW_DERIVATIVES = {}


def register_slow_derivative(name):
    def decorator(fn):
        SLOW_DERIVATIVES[name] = fn
        return fn

    return decorator


@register_slow_derivative("low_pass_filter")
def low_pass_filter(**kwargs) -> float:
    # relaxation of slow dynamics toward fast dynamics
    return -1 / kwargs["tau"] * (kwargs["slow"] - kwargs["fast"])


@register_slow_derivative("use_fast_dynamics")
def use_fast_dynamics(**kwargs) -> float:
    return kwargs["fastdot"]


@register_slow_derivative("integrate_slow_velocity")
def use_slow_dynamics(**kwargs) -> float:
    return kwargs["slowdot"]


@register_slow_derivative("savgol_smoothing")
def use_fast_dynamics2(**kwargs) -> float:
    return kwargs["fastdot"]


def get_slow_derivative(name: str):
    return SLOW_DERIVATIVES.get(name)


class LangevinModel:
    """Langevin model class.

    This class represents a Langevin model used for simulating pedestrian trajectories.
    It contains methods for initializing the model, simulating trajectories, and defining stopping conditions.

    Attributes:
        potential (PiecewisePotential): The piecewise potential object used for modeling.
        params (dict): A dictionary containing the model parameters.

    """

    def __init__(self, potential: PiecewisePotential, params: dict):
        """Initialize Langevin model with parameters.

        Args:
            potential (PiecewisePotential): The piecewise potential object used for modeling.
            params (dict): A dictionary containing the model parameters.

        """
        self.potential = potential
        self.params = params
        self.grid_counts = np.sum(potential.histogram, axis=(2, 3, 4))
        self.heatmap = np.sum(potential.histogram, axis=(2, 3, 4)) / np.sum(potential.histogram)

    def simulate(self, X_0: np.ndarray, t_eval: np.ndarray = np.arange(0, 10, 0.1)) -> np.ndarray:
        """
        Simulates the Langevin model.

        Parameters:
        - X_0: Initial state of the system as a numpy array.
        - t_eval: Time points at which to evaluate the solution. Defaults to np.arange(0, 10, 0.1).

        Returns:
        - The simulated trajectory of the system as a numpy array.
        """
        return sdeint.itoSRI2(self.modelxy, self.Noise, y0=X_0, tspan=t_eval)

    def modelxy(self, X_0: np.ndarray, t) -> np.ndarray:
        """
        Calculate the derivatives of the Langevin model for the given state variables.

        Args:
            X_0 (np.ndarray): Array of initial state variables [xf, yf, uf, vf, xs, ys, us, vs].
            t: Time parameter (not used in this method).

        Returns:
            np.ndarray: Array of derivatives [uf, vf, ufdot, vfdot, xsdot, ysdot, usdot, vsdot].
        """
        xf, yf, uf, vf, xs, ys, us, vs = X_0

        # check stopping condition
        stop_condition = self.params.simulation.stop_condition
        if self.stop_condition(xf, yf, stop_condition) or np.isnan(xs):
            return np.zeros(len(X_0)) * np.nan  # terminate simulation

        rs, thetas = cart2pol(us, vs)
        k = 2
        X_vals = [xs, ys, rs, thetas, k]
        X_indx = get_grid_indices(self.potential, X_vals)
        xmean, xvar, ymean, yvar, umean, uvar, vmean, vvar = self.potential.fit_params[
            X_indx[0], X_indx[1], X_indx[2], X_indx[3], X_indx[4], :
        ]

        # determine potential energy contributions
        V_x = self.potential.curvature_x[X_indx[0], X_indx[1], X_indx[2], X_indx[3], X_indx[4]] * (xf - xmean)
        V_y = self.potential.curvature_y[X_indx[0], X_indx[1], X_indx[2], X_indx[3], X_indx[4]] * (yf - ymean)
        V_u = self.potential.curvature_u[X_indx[0], X_indx[1], X_indx[2], X_indx[3], X_indx[4]] * (uf - umean)
        V_v = self.potential.curvature_v[X_indx[0], X_indx[1], X_indx[2], X_indx[3], X_indx[4]] * (vf - vmean)

        # acceleration fast modes (-grad V, random noise excluded)
        ufdot = -V_x - V_u
        vfdot = -V_y - V_v

        taux = self.params.model.taux
        tauu = self.params.model.tauu
        slow_position_algorithm = self.params.model.slow_positions_algorithm
        slow_velocities_algorithm = self.params.model.slow_velocities_algorithm

        xsdot = get_slow_derivative(slow_position_algorithm)(tau=taux, slow=xs, fast=xf, slowdot=us, fastdot=uf)
        ysdot = get_slow_derivative(slow_position_algorithm)(tau=taux, slow=ys, fast=yf, slowdot=vs, fastdot=vf)

        usdot = get_slow_derivative(slow_velocities_algorithm)(tau=tauu, slow=us, fast=uf, fastdot=ufdot)
        vsdot = get_slow_derivative(slow_velocities_algorithm)(tau=tauu, slow=vs, fast=vf, fastdot=vfdot)

        # return derivatives
        return np.array([uf, vf, ufdot, vfdot, xsdot, ysdot, usdot, vsdot])

    def Noise(self, X_0, t) -> np.ndarray:
        """Return noise matrix.

        Returns:
            numpy.ndarray: A diagonal matrix representing the noise matrix. The diagonal elements
            correspond to the independent driving Wiener processes.

        """
        return np.diag([0.0, 0.0, self.params.model.sigma, self.params.model.sigma, 0.0, 0.0, 0.0, 0.0])

    def stop_condition(self, xf: float, yf: float, stop_condition: float) -> bool:
        """
        Custom stopping condition to terminate a trajectory when the potential goes to infinity.

        Parameters:
        - xf (float): The x-coordinate of the pedestrian.
        - yf (float): The y-coordinate of the pedestrian.
        - stop_condition (float): The threshold value for the stopping condition.

        Returns:
        - bool: A boolean indicating whether the stopping condition has been met.

        """
        c0 = xf < self.params.grid.x.min
        c1 = xf > self.params.grid.x.max
        c2 = yf < self.params.grid.y.min
        c3 = yf > self.params.grid.y.max
        grid_index_x = digitize_values_to_grid(xf, self.potential.bins["x"])
        grid_index_y = digitize_values_to_grid(yf, self.potential.bins["y"])
        c4 = self.grid_counts[grid_index_x, grid_index_y] < stop_condition
        return any([c0, c1, c2, c3, c4])
