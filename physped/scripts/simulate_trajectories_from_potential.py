import logging

import hydra

from physped.processing_pipelines import simulate_from_potential
from physped.utils.config_utils import (
    register_new_resolvers,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../conf", config_name="config"
)
def main(config):
    simulate_from_potential(config)


if __name__ == "__main__":
    register_new_resolvers()
    main()
