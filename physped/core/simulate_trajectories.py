# """Class for different validation settings."""
# import pandas as pd
# import numpy as np
# import os
# import json
# import glob
# import readers as read
# from langevin import Langevin
# from discretize_grid import DiscreteGrid
# from scipy.special import rel_entr, kl_div
# import numpy.ma as ma
# import sys
# import random
# import pickle
# import logging
# from validation import Validation, read_validation_model, get_bin_middle, read_model_from_path
# import customlogging as cl

# log = logging.getLogger('mylog')

# if __name__ == "__main__":
#     log_params = {
#         'level': 'INFO',
#         'display': 'term'
#     }
#     cl.generate_logger(params = log_params)

#     model_path = sys.argv[1]
#     log.info(
#         "Model path: \n"
#         f"{model_path}"
#     )

#     model = read_model_from_path(model_path)
#     # log.info('Model read.')

#     log.info(f'Simulating {model.params["simulation"]["Ntrajectories"]} trajectories...')
#     model.simulate_trajectories()


#     observables = [
#         'xf', 'yf',
#         'uf', 'vf',
#         'rf', 'thetaf'
#     ]
#     for observable in observables:
#         model.create_histogram(observable, traj_type = 'raw')
#         model.create_histogram(observable, traj_type = 'sim')
#         model.compute_KL_divergence(observable)
#     log.info(f'All histograms and KL divergences computed.')
