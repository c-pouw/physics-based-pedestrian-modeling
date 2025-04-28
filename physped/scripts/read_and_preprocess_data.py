from pathlib import Path

import hydra

from physped.processing_pipelines import read_and_preprocess_data
from physped.utils.config_utils import (
    register_new_resolvers,
)

CONFIG = Path.cwd() / "physped" / "conf"


@hydra.main(version_base=None, config_path=str(CONFIG), config_name="config")
def main(config):
    read_and_preprocess_data(config)


if __name__ == "__main__":
    register_new_resolvers()
    main()
