import logging
import os

import hydra

from physped.io.readers import read_grid_bins

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf")
def test_hydra(cfg):
    # Read parameters
    # params = pp.read_parameter_file(name)
    print(f"Current working directory : {os.getcwd()}")
    print(cfg.params)
    grid_bins = read_grid_bins(cfg.params.grid_name)
    print(dict(grid_bins))
    # print(HydraConfig.get().output_subdir)
    return


if __name__ == "__main__":
    test_hydra()
