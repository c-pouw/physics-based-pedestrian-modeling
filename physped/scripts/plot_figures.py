from pathlib import Path

import hydra
import matplotlib.pyplot as plt

from physped.processing_pipelines import plot_figures
from physped.utils.config_utils import (
    register_new_resolvers,
)

CONFIG = Path.cwd() / "physped" / "conf"


@hydra.main(version_base=None, config_path=str(CONFIG), config_name="config")
def main(config):
    plt.style.use(Path(config.root_dir) / config.plot_style)
    plot_figures(config)


if __name__ == "__main__":
    register_new_resolvers()
    main()
