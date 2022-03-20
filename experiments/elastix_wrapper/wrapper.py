from pathlib import Path
import os
import shutil
import subprocess
import logging

from typing import Any, Dict

import wandb
from elastix_wrapper import TimeoutException, time_limit
from elastix_wrapper.parameters import Collection, Parameters
from elastix_wrapper.watchdog import Watchdog

ELASTIX = os.environ.get("ELASTIX_EXECUTABLE")
logger = logging.getLogger("Wrapper")

def run(params: Parameters, run_dir: Path, watch: bool = True) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    params_file = params.write(run_dir)
    out_dir = run_dir.joinpath(Path("out"))
    os.mkdir(out_dir)

    wd = Watchdog(out_dir, params["NumberOfResolutions"])
    if watch:
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

    if watch:
        wandb.save(str((run_dir / "*").resolve()),
                   base_path=str(run_dir.parents[0].resolve()))
        wandb.save(str((run_dir / "out" / "TransformParameters*").resolve()),
                   base_path=str(run_dir.parents[0].resolve()))
        wandb_dir = Path(wandb.run.dir)
        wandb.finish()
        shutil.rmtree(run_dir.absolute())
        shutil.rmtree(wandb_dir.parent.absolute())

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
            os.environ["OMP_NUM_THREADS"]
        ]
        if params.fixedmask_path:
            args += ["-fMask", str(params.fixedmask_path)]
        subprocess.run(
            args,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )


if __name__ == "__main__":
    params = Parameters.from_base(write_img=True, sampler="Full", mesh_size=8).gomea().stopping_criteria(50).instance(Collection.EXAMPLES, 1)
    run(params, Path("output/" + str(params)), False)
