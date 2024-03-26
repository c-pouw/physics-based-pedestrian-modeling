"""Langevin model class."""

import logging

import numpy as np
import sdeint

from physped.core.discrete_grid import PiecewisePotential
from physped.core.functions_to_discretize_grid import get_grid_index
from physped.utils.functions import cart2pol, digitize_values_to_grid

log = logging.getLogger(__name__)


class LangevinModel:
    """Langevin model class.

    This class represents a Langevin model used for simulating pedestrian trajectories.
    It contains methods for initializing the model, simulating trajectories, and defining stopping conditions.

    Attributes:
        grids (DiscreteGrid): The discrete grid object used for modeling.
        params (dict): A dictionary containing the model parameters.
        grid_counts (ndarray): The counts of grid cells visited during simulation.

    """

    def __init__(self, grids: PiecewisePotential, params: dict):
        """Initialize Langevin model with parameters.

        Args:
            grids (DiscreteGrid): The discrete grid object used for modeling.
            params (dict): A dictionary containing the model parameters.

        """
        self.grids = grids
        self.params = params
        self.grid_counts = np.sum(grids.histogram, axis=(2, 3, 4))
        self.heatmap = np.sum(grids.histogram, axis=(2, 3, 4)) / np.sum(grids.histogram)

    def modelxy(self, X_0: np.ndarray, t) -> np.ndarray:
        """
        Given state z=(xf, yf, ..., us, vs), returns the derivatives dz/dt (excluding random noise).

        Parameters:
        - X_0: Initial state vector containing the values of xf, yf, uf, vf, xs, ys, us, vs.

        Returns:
        - dz/dt: Derivatives of the state vector (uf, vf, ufdot, vfdot, xsdot, ysdot, usdot, vsdot).

        """
        # Can we precompute certain quantities to make the processing faster?
        xf, yf, uf, vf, xs, ys, us, vs = X_0
        # check stopping condition
        # Either position out of domain or position in unexplored grid cell
        # stop_condition = self.params.get("stop_condition", 0.000001)
        stop_condition = self.params.simulation.stop_condition
        if self.stop_condition(xf, yf, stop_condition) or np.isnan(xs):
            return np.zeros(len(X_0)) * np.nan  # terminate simulation

        rs, thetas = cart2pol(us, vs)
        k = 2
        X_vals = [xs, ys, rs, thetas, k]
        X_indx = get_grid_index(self.grids, X_vals)
        xmean, xvar, ymean, yvar, umean, uvar, vmean, vvar = self.grids.fit_params[
            X_indx[0], X_indx[1], X_indx[2], X_indx[3], X_indx[4], :
        ]

        # determine potential energy contributions
        # V_x = uvar/xvar*(xf - xmean)
        # V_y = vvar/yvar*(yf - ymean)
        # V_u = self.params['sigma']**2/(2*uvar)*(uf - umean)
        # V_v = self.params['sigma']**2/(2*vvar)*(vf - vmean)

        # determine potential energy contributions
        if xvar == 0:
            V_x = 0
        else:
            V_x = uvar / xvar * (xf - xmean)
        if yvar == 0:
            V_y = 0
        else:
            V_y = vvar / yvar * (yf - ymean)
        if uvar == 0:
            V_u = 0
        else:
            V_u = self.params["sigma"] ** 2 / (2 * uvar) * (uf - umean)
        if vvar == 0:
            V_v = 0
        else:
            V_v = self.params["sigma"] ** 2 / (2 * vvar) * (vf - vmean)

        # acceleration fast modes (-grad V, random noise excluded)
        ufdot = -V_x - V_u
        vfdot = -V_y - V_v

        # relaxation of slow modes toward fast modes
        xsdot = -1 / self.params["taux"] * (xs - xf)
        ysdot = -1 / self.params["taux"] * (ys - yf)
        usdot = -1 / self.params["tauu"] * (us - uf)
        vsdot = -1 / self.params["tauu"] * (vs - vf)
        # k += 1
        # return derivatives
        return np.array([uf, vf, ufdot, vfdot, xsdot, ysdot, usdot, vsdot])

    def simulate(self, X_0: np.ndarray, t_eval: np.ndarray = np.arange(0, 10, 0.1)) -> np.ndarray:
        return sdeint.itoSRI2(self.modelxy, self.Noise, y0=X_0, tspan=t_eval)

    def Noise(self, X_0, t) -> np.ndarray:
        """Return noise matrix.

        Returns:
            numpy.ndarray: A diagonal matrix representing the noise matrix. The diagonal elements
            correspond to the independent driving Wiener processes.

        """
        return np.diag([0.0, 0.0, self.params["sigma"], self.params["sigma"], 0.0, 0.0, 0.0, 0.0])

    def stop_condition(self, xf: float, yf: float, stop_condition: float) -> bool:
        """
        Customize stopping condition.

        Parameters:
        - xf (float): The x-coordinate of the pedestrian's final position.
        - yf (float): The y-coordinate of the pedestrian's final position.
        - stop_condition (float): The threshold value for the stopping condition.

        Returns:
        - bool: A boolean indicating whether the stopping condition has been met.

        """
        grid_index_x = digitize_values_to_grid(xf, self.grids.bins["x"])
        grid_index_y = digitize_values_to_grid(yf, self.grids.bins["y"])
        return self.grid_counts[grid_index_x, grid_index_y] < stop_condition
