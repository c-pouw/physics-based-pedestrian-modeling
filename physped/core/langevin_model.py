"""Langevin model class."""

import logging

import numpy as np
import sdeint

from physped.core.parametrize_potential import get_grid_indices
from physped.core.piecewise_potential import PiecewisePotential
from physped.utils.functions import cartesian_to_polar_coordinates, periodic_angular_conditions

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

        Returns:
            np.ndarray: Array of derivatives [xfdot, yfdot, ufdot, vfdot, xsdot, ysdot, usdot, vsdot].
        """
        xf, yf, uf, vf, xs, ys, us, vs, t, k, Pid = state
        xfdot = uf
        yfdot = vf
        k = np.max([k, 1])
        dt = 1
        dk = self.params.fps

        if np.all(np.isnan(state)) or np.all(np.isinf(state)):
            return np.zeros(len(state))

        if self.particle_outside_grid(xf, yf):
            log.info("Pid %s: left the grid at t = %.2f s", int(Pid), t)
            return np.repeat(np.inf, len(state))

        rs, thetas = cartesian_to_polar_coordinates(us, vs)
        thetas = periodic_angular_conditions(thetas, self.params.grid.bins["theta"])
        slow_state = [xs, ys, rs, thetas, k]
        slow_state_index = get_grid_indices(self.potential, slow_state)

        xmean, ymean, umean, vmean = self.potential.parametrization[
            slow_state_index[0], slow_state_index[1], slow_state_index[2], slow_state_index[3], slow_state_index[4], :, 0
        ]
        beta_x, beta_y, beta_u, beta_v = self.potential.parametrization[
            slow_state_index[0], slow_state_index[1], slow_state_index[2], slow_state_index[3], slow_state_index[4], :, 1
        ]

        if np.all(np.isnan([xmean, ymean, umean, vmean, beta_x, beta_y, beta_u, beta_v])):
            # log.warning("Pid %s: reached hole in the potential at t = %.2f s", int(Pid), t)
            # TODO : Find a fix to handle holes in the potential e.g. coarse graining, closest neighbour

            return np.zeros(len(state)) * np.nan

        V_x = beta_x * (xf - xmean)
        V_y = beta_y * (yf - ymean)

        V_u = beta_u * (uf - umean)
        V_v = beta_v * (vf - vmean)

        # acceleration fast modes (-grad V, random noise excluded)
        ufdot = -V_x - V_u
        vfdot = -V_y - V_v

        slow_position_derivative_algorithm = get_slow_derivative(self.params.model.slow_positions_algorithm)
        slow_velocity_derivative_algorithm = get_slow_derivative(self.params.model.slow_velocities_algorithm)
        xsdot = slow_position_derivative_algorithm(
            tau=self.params.model.taux, slow=xs, fast=xf, slowdot=us, fastdot=uf, dt=self.params.model.dt
        )
        ysdot = slow_position_derivative_algorithm(
            tau=self.params.model.taux, slow=ys, fast=yf, slowdot=vs, fastdot=vf, dt=self.params.model.dt
        )
        usdot = slow_velocity_derivative_algorithm(
            tau=self.params.model.tauu, slow=us, fast=uf, fastdot=ufdot, dt=self.params.model.dt
        )
        vsdot = slow_velocity_derivative_algorithm(
            tau=self.params.model.tauu, slow=vs, fast=vf, fastdot=vfdot, dt=self.params.model.dt
        )

        return np.array([xfdot, yfdot, ufdot, vfdot, xsdot, ysdot, usdot, vsdot, dt, dk, 0])

    def noise(self, X_0, t) -> np.ndarray:
        """Return noise matrix.

        Returns:
            numpy.ndarray: A diagonal matrix representing the noise matrix. The diagonal elements
            correspond to the independent driving Wiener processes.

        """
        noise = self.params.model.sigma
        return np.diag([0.0, 0.0, noise, noise, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def particle_outside_grid(self, xf: float, yf: float) -> bool:
        """Check if particle is outside the grid.

        Args:
            xf: The x-coordinate of the particle.
            yf: The y-coordinate of the particle.

        Returns:
            Whether the particle is outside the grid.
        """
        c0 = xf < self.params.grid.x.min
        c1 = xf > self.params.grid.x.max
        c2 = yf < self.params.grid.y.min
        c3 = yf > self.params.grid.y.max
        return any([c0, c1, c2, c3])
