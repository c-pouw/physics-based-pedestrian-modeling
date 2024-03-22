"""Infer force fields from trajectories."""

import logging
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import norm
from physped.utils.functions import (
    weighted_mean_of_two_matrices,
    digitize_values_to_grid,
    pol2cart,
)
from physped.core.discrete_grid import DiscreteGrid

log = logging.getLogger(__name__)


def trajectories_to_grid(trajectories: pd.DataFrame, grid_bins) -> DiscreteGrid:
    """
    Convert trajectories to a grid of histograms and parameters.

    Parameters:
    - trajectories (pd.DataFrame): A DataFrame of trajectories.
    - grid_vals (dict): A dictionary of grid values for each dimension.

    Returns:
    - A dictionary of DiscreteGrid objects for storing histograms and parameters.
    """
    # grids = initialize_grids(grid_vals)
    grids = DiscreteGrid(grid_bins)
    trajectories = digitize_trajectories_to_grid(grids.bins, trajectories)
    grids.histogram = add_trajectories_to_histogram(grids.histogram, trajectories, "fast_grid_indices")
    grids.histogram_slow = add_trajectories_to_histogram(grids.histogram_slow, trajectories, "slow_grid_indices")
    # trajectory_origins = trajectories.groupby('Pid')['fast_grid_indices'].first().reset_index()
    grids.fit_params = fit_trajectories_on_grid(grids.fit_params, trajectories)
    log.info("Finished converting trajectories to grid.")
    return grids


def accumulate_grids(cummulative_grids: DiscreteGrid, grids_to_add: DiscreteGrid) -> DiscreteGrid:
    """
    Accumulate grids by taking a weighted mean of the fit parameters.

    Parameters:
    - cummulative_grids (DiscreteGrid): The cumulative grids to add to.
    - grids_to_add (DiscreteGrid): The grids to add to the cumulative grids.

    Returns:
    - The updated cumulative grids.
    """
    import copy

    for p in range(cummulative_grids.no_fit_params):  # Loop over all fit parameters
        # accumulate fit parameters
        cummulative_grids.fit_params[:, :, :, :, :, p] = weighted_mean_of_two_matrices(
            a=copy.deepcopy(cummulative_grids.fit_params[:, :, :, :, :, p]),
            aC=copy.deepcopy(cummulative_grids.histogram),
            b=copy.deepcopy(grids_to_add.fit_params[:, :, :, :, :, p]),
            bC=copy.deepcopy(grids_to_add.histogram),
        )
    # accumlate histogram
    cummulative_grids.histogram += grids_to_add.histogram
    cummulative_grids.histogram_slow += grids_to_add.histogram_slow
    return cummulative_grids


def get_slice_of_multidimensional_matrix(a: np.ndarray, slices: List[tuple]) -> np.ndarray:
    """
    Get a slice of a multidimensional NumPy array with periodic boundary conditions.

    Parameters:
    - a (np.ndarray): The input array to slice.
    - slices (List[Tuple[int, int]]): A list of slice ranges for each dimension of the array.

    Returns:
    - The sliced array.
    """
    if any(slice[0] > slice[1] for slice in slices):
        print("Slice values must be ascending.")
    reshape_dimension = (-1,) + (1,) * (len(slices) - 1)
    slices = [
        np.arange(*slice).reshape(np.roll(reshape_dimension, i)) % a.shape[i] for i, slice in enumerate(slices)
    ]
    return a[tuple(slices)]


# def convert_slow_velocities_to_polar(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Convert slow velocities from Cartesian to polar coordinates.

#     Parameters:
#     - df (pd.DataFrame): The input DataFrame containing 'us' and 'vs' columns.

#     Returns:
#     - A new DataFrame with 'thetas' and 'rs' columns added.
#     """
#     rs, thetas = fu.cart2pol(df['us'], df['vs'])
#     return df.assign(rs=rs, thetas=thetas)


def digitize_trajectories_to_grid(grid_bins, trajectories: pd.DataFrame) -> pd.DataFrame:
    """
    Digitize trajectories to a grid.

    Parameters:
    - grid_bins (): The grid bins to digitize the trajectories to.
    - trajectories (pd.DataFrame): The trajectories to digitize.

    Returns:
    - The trajectories with the digitized grid indices.
    """
    indices = {}
    for obs, dynamics in [(obs, dynamics) for obs in grid_bins.keys() for dynamics in ["f", "s"]]:
        if obs == "k":
            dobs = obs
        else:
            dobs = obs + dynamics
        inds = digitize_values_to_grid(trajectories[dobs], grid_bins[obs])
        indices[dobs] = inds

    indices["thetaf"] = np.where(indices["rf"] == 0, 0, indices["thetaf"])
    indices["thetas"] = np.where(indices["rs"] == 0, 0, indices["thetas"])

    trajectories["fast_grid_indices"] = list(
        zip(indices["xf"], indices["yf"], indices["rf"], indices["thetaf"], indices["k"])
    )
    trajectories["slow_grid_indices"] = list(
        zip(indices["xs"], indices["ys"], indices["rs"], indices["thetas"], indices["k"])
    )
    return trajectories


def fit_fast_modes(group: pd.DataFrame) -> list:
    """
    Fits normal distribution to a group of data points and returns fitting parameters.

    Parameters:
    - group (pd.DataFrame): The group of data points to fit the normal distribution to.

    Returns:
    - A list of fitting parameters.
    """
    if len(group) < 10:
        return None  # todo: Can we do better than this?
    fit_func = norm.fit  # todo: Try other fitting functions
    params = []
    for i in ["xf", "yf", "uf", "vf"]:
        # for i in ["xf", "yf", "rf", "thetaf"]: # todo change to polar coordinate fitting
        mu, std = fit_func(group[i])  # fit normal distribution to fast modes
        params += [mu, std**2]  # store mean and variance of normal distribution
    return params


def fit_trajectories_on_grid(param_grid, trajectories: pd.DataFrame):
    """
    Fit trajectories to the parameter grid.

    Parameters:
    - param_grid (): The parameter grid to fit the trajectories on.
    - trajectories (pd.DataFrame): The trajectories to fit.

    Returns:
    - The grid with fit parameters.
    """
    fit_params = trajectories.groupby("slow_grid_indices").apply(fit_fast_modes).dropna().to_dict()
    for key, value in fit_params.items():
        param_grid[key] = value
    return param_grid


def add_trajectories_to_histogram(histogram, trajectories: pd.DataFrame, groupbyindex: str) -> np.ndarray:
    """
    Add trajectories to a histogram.

    Parameters:
    - histogram (): The histogram to add the trajectories to.
    - trajectories (pd.DataFrame): The trajectories to add to the histogram.

    Returns:
    - The updated histogram.
    """
    for grid_index, group in trajectories.groupby(groupbyindex):
        histogram[grid_index] += len(group)
    return histogram


def create_grid_bins(grid_vals: dict) -> dict:
    """
    Create bins for a grid.

    Parameters:
    - grid_vals (dict): The values of the grid.

    Returns:
    - A dictionary of bins for each dimension of the grid.
    """
    grid_bins = {}
    if "x" in grid_vals and "y" in grid_vals:
        grid_bins = {key: np.arange(*grid_vals[key]) for key in ["x", "y"]}
    if "r" in grid_vals:
        grid_bins["r"] = np.array(grid_vals["r"])
    if "theta" in grid_vals:
        grid_bins["theta"] = np.arange(-np.pi, np.pi + 0.01, grid_vals["theta"])
    if "k" in grid_vals:
        grid_bins["k"] = np.array(grid_vals["k"])
    return grid_bins


def get_grid_index(grids: DiscreteGrid, X: List[float]) -> np.ndarray:
    """
    Given a point (xs, ys, thetas, rs), returns the grid index of the point.

    Parameters:
    - grid (Grids): The Grids object containing the grid definitions.
    - X (List[float]): A list of Cartesian and polar coordinates.

    Returns:
    - A tuple of grid indices.
    """
    indices = np.array([], dtype=int)
    for val, obs in zip(X, grids.dimensions):
        grid = grids.bins[obs]
        indices = np.append(indices, digitize_values_to_grid(val, grid))

    if indices[2] == 0:
        indices[3] = 0  # merge grid cells for low r_s

    return indices


def digitize_grid_val(value, grid):
    """Return the grid index of a given value."""
    if value > np.max(grid):
        value -= np.max(grid) - np.min(grid)
        grid_idx = np.digitize(value, grid)
        grid_idx += len(grid) - 1
    elif value < np.min(grid):
        value += np.max(grid) - np.min(grid)
        grid_idx = np.digitize(value, grid)
        grid_idx -= len(grid) - 1
    else:
        grid_idx = np.digitize(value, grid)
    return grid_idx


def return_grid_ids(grid, value):
    """Return the grid indices of a given observable and value."""
    # grid = grids.bins[observable]  # TODO Change attribute to 1d Grid
    # grid = self.grid_observable[observable]

    if isinstance(value, int):
        value = float(value)
    if isinstance(value, float):
        grid_id = digitize_grid_val(value, grid)
        grid_idx = [grid_id - 1, grid_id]

    elif (isinstance(value, list)) and (len(value) == 2):
        grid_idx = [
            digitize_grid_val(value[0], grid) - 1,
            digitize_grid_val(value[1], grid),
        ]
    else:
        grid_idx = [0, len(grid) - 1]
    return {
        "grid_idx": grid_idx,
    }


def grid_bounds(bins, observable, values):
    """Return the grid bounds of a given observable and value."""
    # grid = self.grid_observable[observable]
    # grid = grids.bins[observable]
    if observable == "theta":
        grid_sides = [
            bins[values[0] % (len(bins) - 1)],
            bins[values[1] % (len(bins) - 1)],
        ]
    else:
        grid_sides = [bins[values[0]], bins[values[1]]]
    return grid_sides


def make_grid_selection(grids, selection):
    """Make selection."""
    grid_selection = {}
    for observable, value in selection.items():
        # for observable, value in zip(["x", "y", "r", "theta"], [x, y, r, theta, k]):
        grid_selection[observable] = {}
        # grid = grid.grid_observable[observable]
        grid = grids.bins.get(observable)

        if not value:  # if None select full grid
            value = [grid[0], grid[-2]]
        elif isinstance(value, int):  # if int select single value on grid
            value = [float(value), float(value)]
        elif isinstance(value, float):  # if float select single value on grid
            value = [value, value]

        grid_selection[observable]["selection"] = value
        grid_ids = return_grid_ids(grid, value)["grid_idx"]
        grid_selection[observable]["grid_ids"] = grid_ids
        grid_boundaries = grid_bounds(grid, observable, grid_ids)
        grid_selection[observable]["bounds"] = grid_boundaries

        if observable == "theta":
            while grid_boundaries[1] < grid_boundaries[0]:
                grid_boundaries[1] += 2 * np.pi

        grid_selection[observable]["periodic_bounds"] = grid_boundaries
    return grid_selection


def random_uniform_value_in_bin(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Generate random uniform values within each bin.

    Args:
        values (np.ndarray): Array of bin indices.
        bins (np.ndarray): Array of bin edges.

    Returns:
        np.ndarray: Array of random uniform values within each bin.
    """
    left = bins[values]
    right = bins[values + 1]
    return np.random.uniform(left, right)


def sample_from_ndarray(origin_histogram: np.ndarray, N_samples: int = 1) -> np.ndarray:
    """
    Sample origin positions from a heatmap with initial positions.

    Parameters:
    - origin_histogram (np.ndarray): The initial position heatmap.
    - N_samples (int): The number of samples to generate. Default is 1.

    Returns:
    - A tuple of NumPy arrays representing the sampled origin positions.
    """
    flat_origin_histogram = origin_histogram.ravel()
    indices1d = np.random.choice(
        a=range(len(flat_origin_histogram)),
        size=N_samples,
        replace=True,
        p=flat_origin_histogram / np.sum(flat_origin_histogram),
    )
    return np.array(np.unravel_index(indices1d, origin_histogram.shape)).T


def convert_grid_indices_to_coordinates(grids: DiscreteGrid, X_0: np.ndarray) -> np.ndarray:
    """
    Convert grid indices to Cartesian coordinates.

    Parameters:
    - grids (Grids): The Grids object containing the grid definitions.
    - X_0 (List[int]): A list of grid indices.

    Returns:
    - A list of Cartesian coordinates.
    """
    # TODO: Add some noise within the bin.
    # xf, yf, rf, thetaf, k = (grids.bins[dim][X_0[:,i]] for i, dim in enumerate(grids.dimensions))
    xf = random_uniform_value_in_bin(X_0[:, 0], grids.bins["x"])
    yf = random_uniform_value_in_bin(X_0[:, 1], grids.bins["y"])
    rf = random_uniform_value_in_bin(X_0[:, 2], grids.bins["r"])
    thetaf = random_uniform_value_in_bin(X_0[:, 3], grids.bins["theta"])
    k = grids.bins["k"][X_0[:, 4]]

    uf, vf = pol2cart(rf, thetaf)
    return np.array([xf, yf, uf, vf, k]).T


# def create_and_save_grid(name: str):
#     """Create and save trajectories in grid."""
#     # Read parameters and trajectories
#     params = read.read_validation_parameters(name)
#     trajectories = trajectory_reader[name]()

#     # Preprocess trajectories
#     trajectories.rename(columns={"Rstep": "time", "Pid": "Pid"}, inplace=True)
#     trajectories = preprocess_trajectories(trajectories, params)

#     # Cast trajectories to grid
#     grid_bins = create_grid_bins(params["grid"])
#     grids = trajectories_to_grid(trajectories, grid_bins)

#     # Save the grid
#     filepath = create_filepath(parameters=params)
#     create_folder_if_not_exists(folderpath=os.path.dirname(filepath))
#     save_parameters(parameters=params, folderpath=os.path.dirname(filepath))
#     save_discrete_grid(grids, filepath)


# if __name__ == "__main__":
#     name = sys.argv[1]
#     create_and_save_grid(name)


# def initialize_grids(grid_vals: dict) -> dict:
#     """
#     Initialize three DiscreteGrid objects for storing histograms and parameters.

#     Parameters:
#     - grid_vals (dict): A dictionary of grid values for each dimension.

#     Returns:
#     - A dictionary of DiscreteGrid objects for storing histograms and parameters.
#     """
#     grid_bins = dg.create_grid_bins(grid_vals)
#     param_grid_bins = {**grid_bins, 'param': np.arange(0, 9)}

#     return {
#         'params': DiscreteGrid(param_grid_bins),
#         'hist': DiscreteGrid(grid_bins),
#         'origin_hist': DiscreteGrid(grid_bins)
#     }

# def trajectories_to_grid(trajectories: pd.DataFrame, grid_vals: Dict[str, list]) -> Dict[str, DiscreteGrid]:
#     """
#     Convert trajectories to a grid of histograms and parameters.

#     Parameters:
#     - trajectories (pd.DataFrame): A DataFrame of trajectories.
#     - grid_vals (dict): A dictionary of grid values for each dimension.

#     Returns:
#     - A dictionary of DiscreteGrid objects for storing histograms and parameters.
#     """
#     grids = initialize_grids(grid_vals)
#     trajectories = dg.convert_slow_velocities_to_polar(trajectories)
#     trajectories = dg.digitize_trajectories_to_grid(grids['hist'].bins, trajectories)
#     grids['hist'].grid = dg.add_trajectories_to_histogram(grids['hist'].grid, trajectories)
#     trajectory_origins = trajectories.groupby('Pid')['fast_grid_indices'].first().reset_index()
#     grids['origin_hist'].grid = dg.add_trajectories_to_histogram(
#         grids['origin_hist'].grid, trajectory_origins
#         )
#     grids['params'].grid = dg.fit_trajectories_on_grid(
#         grids['params'].grid, trajectories
#         )
#     return grids

# def main(files):
#     """Process trajectories to force fields."""
#     ## Read parameters
#     grid_vals = val.params['grid']
#     for file in files:
#         trajectories = read_trajectories(file)
#         grids_sub = trajectories_to_grid(trajectories, grid_vals)
#         if 'grids_glob' not in locals():
#             print('Initialize global grids')
#             grids_glob = grids_sub.copy()
#         else:
#             print('Update global grids')
#             for p in range(8):
#                 grids_glob['params'].grid[:,:,:,:,p] = \
#                     fu.weighted_matrix_mean(
#                         a = grids_glob['params'].grid[:,:,:,:,p],
#                         aC = grids_glob['hist'].grid,
#                         b = grids_sub['params'].grid[:,:,:,:,p],
#                         bC = grids_sub['hist'].grid
#                     )
#             grids_glob['hist'].grid += grids_sub['hist'].grid
#             grids_glob['origin_hist'].grid += grids_sub['origin_hist'].grid


# # ------------------------------------------------------------------

# def digitize_grid_val(value, grid):
#     """Return the grid index of a given value."""
#     if value > np.max(grid):
#         value -= (np.max(grid) - np.min(grid))
#         grid_idx = np.digitize(value, grid)
#         grid_idx += len(grid)-1
#     elif value < np.min(grid):
#         value += (np.max(grid) - np.min(grid))
#         grid_idx = np.digitize(value, grid)
#         grid_idx -= len(grid)-1
#     else:
#         grid_idx = np.digitize(value, grid)
#     return grid_idx

# class DiscreteGrid_legacy():
#     def __init__(self, df, xgrid, ygrid, rgrid, thetagrid):
#         self.xgrid = xgrid # arrays with bin edges
#         self.ygrid = ygrid
#         self.thetagrid = thetagrid
#         self.rgrid = rgrid
#         self.grid_observable = {
#             'x': xgrid,
#             'y': ygrid,
#             'r': rgrid,
#             'theta': self.thetagrid
#         }
#         self.df = df
#         # dataframe containing measurement data (including fast and slow modes and 'Pid')

#         self.grid = np.meshgrid(xgrid, ygrid, rgrid, thetagrid, indexing='ij')
#         # meshgrid with bin edges

#         # Calculate the centers of each grid cell
#         self.xgrid_middle = fu.get_bin_middle(xgrid)
#         self.ygrid_middle = fu.get_bin_middle(ygrid)
#         self.rgrid_middle = fu.get_bin_middle(rgrid)
#         self.thetagrid_middle = fu.get_bin_middle(thetagrid)
#         x_centers = self.xgrid_middle
#         y_centers = self.ygrid_middle
#         r_centers = self.rgrid_middle
#         theta_centers = self.thetagrid_middle

#         # Create meshgrid with center coordinates
#         self.centers_grid = np.meshgrid(
#             self.xgrid_middle, self.ygrid_middle,
#             self.rgrid_middle, self.thetagrid_middle, indexing='ij'
#         )

#         # Combine the centers into a 4D array
#         self.centers = np.stack(self.centers_grid, axis=-1)

#         # Create parameter array with dimension
#         # (#x-cells, #y-cells, #theta-cells, #r-cells, number of parameters)
#         self.params = np.zeros(
#             (len(x_centers), len(y_centers), len(r_centers), len(theta_centers), 8)
#         )*np.nan # set to nans
#         self.params_df = self.params.copy()

#         self.nd_grid_histogram = np.zeros(
#             (len(x_centers), len(y_centers), len(r_centers), len(theta_centers))
#         )
#         self.nd_grid_histogram_df = self.nd_grid_histogram.copy()

#         self.origin_histogram = np.zeros(
#             (len(x_centers), len(y_centers), len(r_centers), len(theta_centers))
#         )
#         self.origin_histogram_df = self.origin_histogram.copy()

#         x_slow_indices = fu.digitize_values_to_grid(df['xs'], xgrid)
#         y_slow_indices = fu.digitize_values_to_grid(df['ys'], ygrid)

#         x_fast_indices = fu.digitize_values_to_grid(df['xf'], xgrid)
#         y_fast_indices = fu.digitize_values_to_grid(df['yf'], ygrid)

#         # determine polar coordinate representation for (us, vs)
#         thetas, rs = fu.cart2pol(df['us'], df['vs'])
#         df['thetas'] = thetas
#         df['rs'] = rs

#         r_slow_indices = fu.digitize_values_to_grid(df['rs'], rgrid)
#         theta_slow_indices = fu.digitize_values_to_grid(df['thetas'], thetagrid)
#         theta_slow_indices = np.where(r_slow_indices==0, 0, theta_slow_indices)

#         r_fast_indices = fu.digitize_values_to_grid(df['rf'], rgrid)
#         theta_fast_indices = fu.digitize_values_to_grid(df['thetaf'], thetagrid)
#         theta_fast_indices = np.where(r_fast_indices==0, 0, theta_fast_indices)

#         df['slow_grid_indices'] = list(zip(
#             x_slow_indices, y_slow_indices, r_slow_indices, theta_slow_indices
#         ))
#         df['fast_grid_indices'] = list(zip(
#             x_fast_indices, y_fast_indices, r_fast_indices, theta_fast_indices
#         ))

#         self.create_param_grid(df) # Fit the parameters of the df

#         self.create_origin_histogram_df(df)
#         self.create_ndimensional_grid_histogram_df(df)

#         self.params = self.params_df.copy()

#         self.origin_histogram += self.origin_histogram_df
#         self.nd_grid_histogram += self.nd_grid_histogram_df

#     def add_files(file):
#         df = read_trajectories(file)
#         self.add_trajectories(df)
#         #read df from file using the right read function

#     def add_trajectories(self, df):
#         """Add trajectories to discretegrid."""
#         # thetas, rs = fu.cart2pol(df['us'], df['vs'])
#         # df['thetas'] = thetas
#         # df['rs'] = rs
#         # hist, edges = np.histogramdd(
#         #     df[['xs', 'ys', 'rs', 'thetas']].values,
#         #     bins=[self.xgrid, self.ygrid,
#         #           self.rgrid, self.thetagrid]
#         # )
#         #self.eval = (hist>10)  # only evaluate if grid cell has more than 10 data points

#         # create new histograms
#         self.create_origin_histogram_df(df)
#         self.create_ndimensional_grid_histogram_df(df)

#         # Accumulate fit params
#         self.create_param_grid(df)
#         self.update_fit_params()

#         # # Accumulate histograms
#         self.origin_histogram += self.origin_histogram_df
#         self.nd_grid_histogram += self.nd_grid_histogram_df

#     def fit_fun(self, group):
#         """Fits normal distribution to a group of data points, returns fitting parameters"""
#         if len(group) > 10:  # only fit when group contains >10 data points

#             params = []
#             for i in ['xf','yf','uf','vf']:
#                 mufit, stdfit = norm.fit(group[i])  # fit normal distribution to fast modes
#                 params += [mufit, stdfit**2]  # store mean and variance of normal distribution
#             return params
#         else:
#             return None

#     def create_param_grid(self, df):
#         """Applies fitting procedure to all explored grid cells, stores fitting parameters in self.params"""

#         # group data by grid cell and apply fitting procedure to every group
#         params_dict = dict(df.groupby('slow_grid_indices').apply(
#             lambda group: self.fit_fun(group)).dropna()
#                            )

#         # store all fitting parameters in self.params
#         for i, grid_index in enumerate(params_dict):
#             self.params_df[grid_index] = params_dict[grid_index]

#     def update_fit_params(self):
#         log.info('start param update')

#         for p in range(8):
#             a = self.params[:,:,:,:,p]
#             b = self.params_df[:,:,:,:,p]
#             aC = self.nd_grid_histogram
#             bC = self.nd_grid_histogram_df

#             weighted_mean = fu.weighted_matrix_mean(a, aC, b, bC)
#             self.params[:,:,:,:,p] = weighted_mean
#             log.info('params updated')


#     def get_grid_index(self, point):
#         """Given a point (xs, ys, thetas, rs), returns the grid index of the point"""

#         # max(0,index) and min(index, len(grid)-2) ensures that index is not outside domain
#         x_index = max(0, min(np.digitize(point[0], self.xgrid) - 1, len(self.xgrid) - 2))
#         y_index = max(0, min(np.digitize(point[1], self.ygrid) - 1, len(self.ygrid) - 2))
#         r_index = max(0, min(np.digitize(point[3], self.rgrid) - 1, len(self.rgrid) - 2))
#         theta_index = max(0, min(np.digitize(point[2], self.thetagrid) - 1, len(self.thetagrid) - 2))

#         if r_index == 0:
#             theta_index = 0  # merge grid cells for low r_s

#         return (x_index, y_index, r_index, theta_index)

#     def retrieve_params(self, X):
#         """Given a point X=(xs, ys, us, vs), returns the fitting parameters of the associated grid cell"""
#         xs, ys, us, vs = X

#         # convert (us, vs) to polar coordinates (thetas, rs)
#         thetas = np.arctan2(vs,us)
#         rs = np.hypot(us,vs)

#         # determine grid index of (xs, ys, thetas, rs)
#         index = self.get_grid_index((xs,ys,rs,thetas))

#         # return fitting parameters
#         return self.params[index]

#     def return_grid_ids(self, observable, value):
#         """Return the grid indices of a given observable and value."""
#         grid = self.grid_observable[observable]

#         if isinstance(value, int):
#             value = float(value)
#         if isinstance(value, float):
#             grid_id = digitize_grid_val(value, grid)
#             grid_idx = [
#                 grid_id-1,
#                 grid_id
#             ]

#         elif (isinstance(value, list)) and (len(value) == 2):
#             grid_idx = [
#                 digitize_grid_val(value[0], grid) - 1,
#                 digitize_grid_val(value[1], grid)
#             ]
#         else:
#             grid_idx = [
#                 0,
#                 len(grid) - 1
#             ]
#         return {
#             'grid_idx': grid_idx,
#         }

#     def grid_bounds(self, observable, values):
#         """Return the grid bounds of a given observable and value."""
#         grid = self.grid_observable[observable]
#         if observable == 'theta':
#             grid_sides = [
#                 grid[values[0]%(len(grid)-1)],
#                 grid[values[1]%(len(grid)-1)]
#             ]
#         else:
#             grid_sides = [
#                 grid[values[0]],
#                 grid[values[1]]
#             ]
#         return grid_sides

#     def create_origin_histogram_df(self, df):
#         """Create origin histogram dataframe."""
#         origins = (
#             df
#             .groupby('Pid')
#             .agg({'fast_grid_indices': 'first'})
#             .reset_index()
#         )

#         for grid_index, group in origins.groupby('fast_grid_indices'):
#             self.origin_histogram_df[grid_index] = len(group)

#     def create_ndimensional_grid_histogram_df(self, df):
#         """Create n-dimensional grid histogram dataframe."""
#         for grid_index, group in df.groupby('fast_grid_indices'):
#             self.nd_grid_histogram_df[grid_index] = len(group)
