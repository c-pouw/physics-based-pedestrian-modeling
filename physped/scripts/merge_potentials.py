import logging
from pathlib import Path

import hydra
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# from physped.core.parametrize_potential import (
#     learn_potential_from_trajectories,
# )
# from physped.core.pedestrian_simulator import simulate_pedestrians
# from physped.core.slow_dynamics import compute_slow_dynamics
from physped.io.readers import (
    read_piecewise_potential_from_file,
)

# from physped.io.writers import save_piecewise_potential, save_trajectories
# from physped.preprocessing.trajectories import preprocess_trajectories
# from physped.processing_pipelines import read_and_preprocess_data
from physped.utils.config_utils import (
    log_configuration,
    register_new_resolvers,
)

log = logging.getLogger(__name__)
CONFIG = Path.cwd() / "physped" / "conf"


def validate_potentials(potentials) -> bool:
    """Check if the list with potentials is valid."""
    # n_potentials = len(potentials)
    # lattice_shapes = np.zeros(n_potentials)
    lattice_shapes = []
    histogram_shapes = []
    histogram_slow_shapes = []
    fit_params = []
    parameterization_shapes = []
    for potential in potentials:
        lattice_shapes.append(potential.lattice.shape)
        histogram_shapes.append(potential.histogram.shape)
        histogram_slow_shapes.append(potential.histogram_slow.shape)
        fit_params.append(potential.dist_approximation.fit_parameters)
        parameterization_shapes.append(potential.parameterization.shape)

    # Check if all lattice shapes are identical
    if not all(shape == lattice_shapes[0] for shape in lattice_shapes):
        log.info(lattice_shapes)
        log.error("Lattice shapes do not match")
        return False

    # Check if all histogram shapes are identical
    if not all(shape == histogram_shapes[0] for shape in histogram_shapes):
        log.error("Histogram shapes do not match")
        return False

    # Check if all histogram slow shapes are identical
    if not all(
        shape == histogram_slow_shapes[0] for shape in histogram_slow_shapes
    ):
        log.error("Histogram slow shapes do not match")
        return False

    # Check if all fit parameters are identical
    if not all(param == fit_params[0] for param in fit_params):
        log.error("Fit parameters do not match")
        return False

    # Check if all parameterization shapes are identical
    if not all(
        shape == parameterization_shapes[0]
        for shape in parameterization_shapes
    ):
        log.error("Parameterization shapes do not match")
        return False

    return True


def merge_potentials(config):
    log.info("Start learning potential from multiple files.")
    log_configuration(config)

    filepath1 = Path.cwd() / "potentials" / config.filename.piecewise_potential
    filepaths = [filepath1, filepath1]
    potentials = []
    with logging_redirect_tqdm():
        for filepath in tqdm(
            filepaths, desc="Files processed", total=len(filepaths)
        ):
            logging.info("File %s", filepath)

            potentials.append(read_piecewise_potential_from_file(filepath))

    if validate_potentials(potentials):
        log.info("Potentials validated and ready to be merged.")
    else:
        log.critical("Potentials can not be merged.")
        exit

    for potential in potentials:
        log.info(potential)
        log.info(potential.histogram.shape)
        log.info(potential.histogram.sum())
        log.info(potential.parametrization.shape)

    # save_piecewise_potential(
    #     piecewise_potential,
    #     Path.cwd() / "potentials",
    #     config.filename.piecewise_potential,
    # )


@hydra.main(version_base=None, config_path=str(CONFIG), config_name="config")
def main(config):
    merge_potentials(config)


if __name__ == "__main__":
    register_new_resolvers()
    main()
