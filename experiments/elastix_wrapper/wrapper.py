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
    run_dir.mkdir(parents=True)
    params_file = params.write(run_dir)
    out_dir = run_dir.joinpath(Path("out"))
    os.mkdir(out_dir)

    wd = Watchdog(out_dir, params["NumberOfResolutions"])
    if watch:
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

    logger.info("Run finished successfully.")
    wd.stop()
    wd.join()

    if watch:
        wandb.save(
            str((run_dir / "*").resolve()), base_path=str(run_dir.parents[0].resolve())
        )
        wandb.save(
            str((run_dir / "out" / "TransformParameters*").resolve()),
            base_path=str(run_dir.parents[0].resolve()),
        )
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
            os.environ["OMP_NUM_THREADS"],
        ]
        if params.fixedmask_path:
            args += ["-fMask", str(params.fixedmask_path)]
        subprocess.run(args, check=True)


if __name__ == "__main__":
    sched = [7, 7, 7]
    iterations_g = [20, 30, 50]
    iterations_a = [2000, 2000, 3000]
    params = (
        Parameters.from_base(mesh_size=4, seed=1523, write_img=True)
        .multi_resolution(1, p_sched=sched)
        .gomea()
        .instance(Collection.EMPIRE, 26)
        .stopping_criteria(iterations=iterations_g)
    )
    run(params, Path("output/" + str(params)), False)
