import hydra

from physped.processing_pipelines import (
    sample_and_save_dynamics_from_trajectories,
)
from physped.utils.config_utils import (
    register_new_resolvers,
)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    sample_and_save_dynamics_from_trajectories(config)


if __name__ == "__main__":
    register_new_resolvers()
    main()
