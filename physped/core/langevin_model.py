"""Langevin model class."""

import logging

import numpy as np
import sdeint

from physped.core.functions_to_discretize_grid import get_grid_indices
from physped.core.piecewise_potential import PiecewisePotential
from physped.preprocessing.trajectories import periodic_angular_conditions
from physped.utils.functions import cart2pol

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

    def __init__(self, potential: PiecewisePotential, params: dict, pid: int):
        """Initialize Langevin model with parameters.

        Args:
            potential (PiecewisePotential): The piecewise potential object used for modeling.
            params (dict): A dictionary containing the model parameters.

        """
        self.pid = pid
        self.potential = potential
        self.params = params
        # self.grid_counts = np.sum(potential.histogram, axis=(2, 3, 4))
        # self.heatmap = np.sum(potential.histogram, axis=(2, 3, 4)) / np.sum(potential.histogram)

    def simulate(self, X_0: np.ndarray, t_eval: np.ndarray = np.arange(0, 10, 0.1)) -> np.ndarray:
        """
        Simulates the Langevin model.

        Parameters:
        - X_0: Initial state of the system as a numpy array.
        - t_eval: Time points at which to evaluate the solution. Defaults to np.arange(0, 10, 0.1).

        Returns:
        - The simulated trajectory of the system as a numpy array.
        """
        return sdeint.itoSRI2(self.modelxy, self.noise, y0=X_0, tspan=t_eval)

    def modelxy(self, state: np.ndarray, t) -> np.ndarray:
        """
        Calculate the derivatives of the Langevin model for the given state variables.

        Args:
            X_0 (np.ndarray): Array of initial state variables [xf, yf, uf, vf, xs, ys, us, vs].
            t: Time parameter (not used in this method).

        Returns:
            np.ndarray: Array of derivatives [uf, vf, ufdot, vfdot, xsdot, ysdot, usdot, vsdot].
        """
        xf, yf, uf, vf, xs, ys, us, vs = state

        if np.all(np.isnan(state)):
            return np.zeros(len(state))

        if self.particle_outside_grid(xf, yf):
            log.critical("%s: Trajectory outside grid at t = %.2f s", self.pid, t)
            return np.zeros(len(state)) * np.nan

        rs, thetas = cart2pol(us, vs)
        thetas = periodic_angular_conditions(thetas, self.params.grid.bins["theta"])
        k = 2
        slow_state = [xs, ys, rs, thetas, k]
        slow_state_index = get_grid_indices(self.potential, slow_state)
        xmean, xvar, ymean, yvar, umean, uvar, vmean, vvar = self.potential.fit_params[
            slow_state_index[0], slow_state_index[1], slow_state_index[2], slow_state_index[3], slow_state_index[4], :
        ]

        if np.all([np.isnan(x) for x in [xmean, ymean, umean, vmean, xvar, yvar, uvar, vvar]]):
            log.critical("%s: All free parameters nan at t = %.2f s - with state %s", self.pid, t, slow_state_index)
            return np.zeros(len(state)) * np.nan

        beta_x = self.potential.curvature_x[*slow_state_index]
        beta_y = self.potential.curvature_y[*slow_state_index]
        beta_u = self.potential.curvature_u[*slow_state_index]
        beta_v = self.potential.curvature_v[*slow_state_index]

        V_x = beta_x * (xf - xmean)
        V_y = beta_y * (yf - ymean)

        V_u = beta_u * (uf - umean)
        V_v = beta_v * (vf - vmean)

        # acceleration fast modes (-grad V, random noise excluded)
        ufdot = -V_x - V_u
        vfdot = -V_y - V_v

        dt = self.params.model.dt
        slow_position_derivative_algorithm = get_slow_derivative(self.params.model.slow_positions_algorithm)
        slow_velocity_derivative_algorithm = get_slow_derivative(self.params.model.slow_velocities_algorithm)
        xsdot = slow_position_derivative_algorithm(tau=self.params.model.taux, slow=xs, fast=xf, slowdot=us, fastdot=uf, dt=dt)
        ysdot = slow_position_derivative_algorithm(tau=self.params.model.taux, slow=ys, fast=yf, slowdot=vs, fastdot=vf, dt=dt)
        usdot = slow_velocity_derivative_algorithm(tau=self.params.model.tauu, slow=us, fast=uf, fastdot=ufdot, dt=dt)
        vsdot = slow_velocity_derivative_algorithm(tau=self.params.model.tauu, slow=vs, fast=vf, fastdot=vfdot, dt=dt)

        return np.array([uf, vf, ufdot, vfdot, xsdot, ysdot, usdot, vsdot])

    def noise(self, X_0, t) -> np.ndarray:
        """Return noise matrix.

        Returns:
            numpy.ndarray: A diagonal matrix representing the noise matrix. The diagonal elements
            correspond to the independent driving Wiener processes.

        """
        noise = self.params.model.sigma
        return np.diag([0.0, 0.0, noise, noise, 0.0, 0.0, 0.0, 0.0])

    def particle_outside_grid(self, xf: float, yf: float) -> bool:
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
        return any([c0, c1, c2, c3])
