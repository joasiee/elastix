from pathlib import Path
import os
import subprocess
import logging

from typing import Any, Dict
from elastix_wrapper import TimeoutException, time_limit
from elastix_wrapper.parameters import Collection, Parameters
from elastix_wrapper.watchdog import Watchdog

ELASTIX = os.environ.get("ELASTIX_EXECUTABLE")
logger = logging.getLogger("Wrapper")

def run(params: Parameters, run_dir: Path) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    params_file = params.write(run_dir)
    out_dir = run_dir.joinpath(Path("out"))
    os.mkdir(out_dir)

    wd = Watchdog(out_dir, params["NumberOfResolutions"])
    wd.start()

    logger.info(f"Starting elastix for: {str(params)}.")
    try:
        execute_elastix(params_file, out_dir, params)
    except subprocess.CalledProcessError as err:
        logger.error(
            f"Something went wrong while running elastix with params: {str(params)}: {str(err)}")
        return
    except TimeoutException:
        logger.info(
            f"Exceeded time limit of {params['MaxTimeSeconds']} seconds.")
    except KeyboardInterrupt:
        logger.info(f"Run ended prematurely by user.")

    logger.info("Run finished successfully.")
    wd.stop()

def execute_elastix(params_file: Path, out_dir: Path, params: Parameters):
    with time_limit(params["MaxTimeSeconds"]):
        subprocess.run(
            [
                ELASTIX,
                "-p",
                str(params_file),
                "-f",
                str(params.fixed_path),
                "-m",
                str(params.moving_path),
                "-out",
                str(out_dir)
            ],
            check=True
        )

if __name__ == "__main__":
    params = (
        Parameters(mesh_size=8, sampler="Full")
        .gomea()
        .instance(Collection.EXAMPLES, 1)
        .stopping_criteria(iterations=100)
    )
    run(params, Path("output/" + str(params)))
