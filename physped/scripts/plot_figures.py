import hydra

from physped.processing_pipelines import plot_figures
from physped.utils.config_utils import (
    register_new_resolvers,
)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    plot_figures(config)


if __name__ == "__main__":
    register_new_resolvers()
    main()
