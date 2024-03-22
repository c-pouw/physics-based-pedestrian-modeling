"""Langevin model class."""

import logging

import numpy as np
import sdeint

from physped.core.discrete_grid import DiscreteGrid
from physped.core.functions_to_discretize_grid import get_grid_index
from physped.utils.functions import cart2pol, digitize_values_to_grid

log = logging.getLogger(__name__)


class LangevinModel:
    """Langevin model class."""

    def __init__(self, grids: DiscreteGrid, params: dict):
        """Initialize Langevin model with parameters."""
        self.grids = grids
        self.params = params
        self.grid_counts = np.sum(grids.histogram, axis=(2, 3, 4))
        # self.heatmap = np.sum(grids.histogram, axis=(2, 3, 4)) / np.sum(grids.histogram)

    def modelxy(self, X_0, t):
        """Given state z=(xf, yf, ..., us, vs), returns the derivatives dz/dt (excluding random noise)."""
        # TODO: Can we precompute certain quantities to make the processing faster?
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
        xmean, xvar, ymean, yvar, umean, uvar, vmean, vvar = self.grids.fit_params[*X_indx, :]

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

    def simulate(self, X_0: np.ndarray, t_eval: np.ndarray = np.arange(0, 10, 0.1)):
        """Simulate trajectories.

        Args:
            X_0 (np.ndarray): Initial values [xf_i, yf_i, uf_i, vf_i, xs_i, ys_i, us_i, vs_i].
            t_eval (np.ndarray, optional): Time evaluation np.arange(t_i, t_f, timestep). Defaults to np.arange(0, 10, 0.1).

        Returns:
            np.ndarray: Solutions in self.xf, self.yf, ..., self.us, self.vs.
        """
        return sdeint.itoSRI2(self.modelxy, self.Noise, y0=X_0, tspan=t_eval)

    def Noise(self, X_0, t):
        """Return noise matrix."""
        # diagonal, so independent driving Wiener processes
        return np.diag([0.0, 0.0, self.params["sigma"], self.params["sigma"], 0.0, 0.0, 0.0, 0.0])

    def stop_condition(self, xf: float, yf: float, stop_condition: float) -> bool:
        """
        Customize stopping condition.

        Parameters:
        - X_0 (List[float]): A list of initial values for the simulation.

        Returns:
        - A boolean indicating whether the stopping condition has been met.
        """
        # return not (self.grids.bins["x"].min() < xf < self.grids.bins["x"].max()) and (
        #     self.grids.bins["y"].min() < yf < self.grids.bins["y"].max()
        # )
        # print(self.heatmap.shape)
        # print(self.heatmap)
        grid_index_x = digitize_values_to_grid(xf, self.grids.bins["x"])
        grid_index_y = digitize_values_to_grid(yf, self.grids.bins["y"])
        return self.grid_counts[grid_index_x, grid_index_y] < stop_condition
        # print(self.heatmap[grid_index_x, grid_index_y])
        # return self.heatmap[grid_index_x, grid_index_y] < stop_condition

        # xmin, xmax, _ = self.params['grid']['x']
        # ymin, ymax, _ = self.params['grid']['y']
        # # returns True if (xf, yf) is outside the domain
        # return not (xmin < xf < xmax and ymin < yf < ymax)
