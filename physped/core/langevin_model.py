"""Langevin model class."""

import logging

import numpy as np
import sdeint

from physped.core.functions_to_discretize_grid import get_grid_indices
from physped.core.piecewise_potential import PiecewisePotential
from physped.utils.functions import cart2pol, digitize_values_to_grid

log = logging.getLogger(__name__)


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

        # relaxation of slow modes toward fast modes
        xsdot = -1 / self.params.taux * (xs - xf)
        ysdot = -1 / self.params.taux * (ys - yf)
        usdot = -1 / self.params.tauu * (us - uf)
        vsdot = -1 / self.params.tauu * (vs - vf)

        # return derivatives
        return np.array([uf, vf, ufdot, vfdot, xsdot, ysdot, usdot, vsdot])

    def Noise(self, X_0, t) -> np.ndarray:
        """Return noise matrix.

        Returns:
            numpy.ndarray: A diagonal matrix representing the noise matrix. The diagonal elements
            correspond to the independent driving Wiener processes.

        """
        return np.diag([0.0, 0.0, self.params.sigma, self.params.sigma, 0.0, 0.0, 0.0, 0.0])

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
