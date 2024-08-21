from physped.utils.config_utils import initialize_hydra_config


def test_initialize_hydra_config():
    config = initialize_hydra_config("tests")
    assert config.params.env_name == "tests"
