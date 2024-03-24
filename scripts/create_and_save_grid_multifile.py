import logging
from pathlib import Path

import hydra

# import pandas as pd

# from physped.core.functions_to_discretize_grid import accumulate_grids
# from physped.io.readers import read_discrete_grid_from_file, trajectory_reader

log = logging.getLogger(__name__)

# optional_filters = [pp.filter_trajectories_by_velocity]


@hydra.main(version_base=None, config_path="../conf")
def create_and_save_grid(cfg):
    """Create and save trajectories in grid."""
    # Read parameters and trajectories
    folderpath = Path(cfg.params.folder_path)
    print(folderpath)
    # name = cfg.params.env_name
    # params = cfg.params


#     # datelist = pd.date_range(start="2023-10-01", end="2023-10-07", freq="1d")
#     freq = "6h"
#     datehourlist = pd.date_range(start="2023-10-01", end="2023-10-02", freq=freq, inclusive="left")
#     total_grids = None
#     # for date in datelist:
#     for datehour in datehourlist:
#         # trajectories = pp.trajectory_reader[name](date)
#         try:
#             trajectories = trajectory_reader[name](datehour, freq)
#         except ValueError as e:
#             log.warning("Could not read %s: %s", datehour, e)
#             continue

#         if trajectories.empty:
#             log.warning("Skipping empty dataframe %s", datehour)
#             continue

#         trajectories = trajectories.dropna()
#         ## TODO: Replace velocity NaNs with closest value
#         # Preprocess trajectories
#         trajectories.rename(
#             columns={
#                 "date_time_utc": "time",
#                 "tracked_object": "Pid",
#                 "t": "time",
#                 "x_pos": "xf",
#                 "y_pos": "yf",
#                 "x_vel": "uf",
#                 "y_vel": "vf",
#             },
#             inplace=True,
#         )

#         trajectories = preprocess_trajectories(trajectories, params)

#         # Cast trajectories to grid
#         grid_bins = create_grid_bins(params["grid"])
#         grids = trajectories_to_grid(trajectories, grid_bins)
#         if total_grids is None:
#             total_grids = grids
#         else:
#             total_grids = accumulate_grids(total_grids, grids)

#     # Save the grid
#     filepath = pp.create_filepath(params=params)
#     create_folder_if_not_exists(folderpath=filepath.parent)
#     save_parameters(parameters=params, folderpath=filepath.parent)
#     save_discrete_grid(total_grids, filepath)


if __name__ == "__main__":
    create_and_save_grid()
