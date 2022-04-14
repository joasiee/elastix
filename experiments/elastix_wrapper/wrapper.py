from pathlib import Path
import os
import subprocess
import logging

from typing import Any, Dict

from elastix_wrapper import TimeoutException, time_limit
from elastix_wrapper.parameters import Collection, Parameters
from elastix_wrapper.watchdog import SaveStrategyFile, Watchdog

ELASTIX = os.environ.get("ELASTIX_EXECUTABLE")
logger = logging.getLogger("Wrapper")


def run(params: Parameters, run_dir: Path) -> Dict[str, Any]:
    run_dir.mkdir(parents=True)
    params_file = params.write(run_dir)
    out_dir = run_dir.joinpath(Path("out"))
    os.mkdir(out_dir)

    wd = Watchdog(out_dir, params["NumberOfResolutions"])
    wd.set_strategy(SaveStrategyFile(Path()))
    wd.start()

    logger.info(f"Running elastix in: {str(run_dir)}")
    try:
        execute_elastix(params_file, out_dir, params)
    except subprocess.CalledProcessError as err:
        logger.error(
            f"Something went wrong while running elastix at: {str(run_dir)}: {str(err)}"
        )
        return
    except TimeoutException:
        logger.info(f"Exceeded time limit of {params['MaxTimeSeconds']} seconds.")
    except KeyboardInterrupt:
        logger.info(f"Run ended prematurely by user.")

    wd.stop()
    logger.info("Run finished successfully.")



def execute_elastix(params_file: Path, out_dir: Path, params: Parameters):
    with time_limit(params["MaxTimeSeconds"]):
        args = [
            ELASTIX,
            "-p",
            str(params_file),
            "-f",
            str(params.fixed_path),
            "-m",
            str(params.moving_path),
            "-out",
            str(out_dir),
            "-threads",
            os.environ["OMP_NUM_THREADS"],
        ]
        if params.fixedmask_path and params["UseMask"]:
            args += ["-fMask", str(params.fixedmask_path)]

        subprocess.run(args, check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    params = (
        Parameters.from_base(mesh_size=5, sampler="Full")
        .multi_resolution(3)
        .asgd()
        .stopping_criteria(2000)
        .instance(Collection.EXAMPLES, 1)
    )
    run(params, Path("output/" + str(params)))
