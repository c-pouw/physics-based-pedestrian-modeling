from hydra import compose, initialize
from omegaconf import DictConfig

from physped.omegaconf_resolvers import register_new_resolvers


def initialize_hydra_config(env_name: str) -> DictConfig:
    """Function to initialize the Hydra configuration.

    Args:
        env_name: The name of the environment.
            For example: 'single_paths', 'intersecting_paths', 'asdz_pf12', 'asdz_pf34', 'utrecht_pf5'.

    Returns:
        The Hydra configuration.
    """
    with initialize(version_base=None, config_path="../conf"):
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=[
                f"params={env_name}",
            ],
        )
        register_new_resolvers(replace=True)
    return config
