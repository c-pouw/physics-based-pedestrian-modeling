# %%
import logging

import hydra
import numpy as np

import physped as pp


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf")
def create_grid(cfg):
    log.info("Starting script.")
    xbins = np.arange(cfg.grid.x.min, cfg.grid.x.max, cfg.grid.x.step)
    ybins = np.arange(cfg.grid.y.min, cfg.grid.y.max, cfg.grid.y.step)
    rbins = np.arange(cfg.grid.r.min, cfg.grid.r.max, cfg.grid.r.step)
    thetabins = np.linspace(-np.pi, np.pi + 0.01, cfg.grid.theta.chunks)
    kbins = np.array([0, 1, 10**10])
    np.savez(file=f"data/grids/{cfg.grid.name}.npz", x=xbins, y=ybins, r=rbins, theta=thetabins, k=kbins)
    log.info("Script finished.")


if __name__ == "__main__":
    create_grid()
