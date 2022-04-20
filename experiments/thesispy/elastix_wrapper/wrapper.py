from pathlib import Path
import os
import subprocess
import logging

from typing import Any, Dict

from thesispy.elastix_wrapper import TimeoutException, time_limit
from thesispy.elastix_wrapper.parameters import Collection, Parameters
from thesispy.elastix_wrapper.watchdog import SaveStrategy, Watchdog

ELASTIX = os.environ.get("ELASTIX_EXECUTABLE")
logger = logging.getLogger("Wrapper")


def run(
    params: Parameters, run_dir: Path, save_strategy: SaveStrategy = None
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True)
    params_file = params.write(run_dir)
    out_dir = run_dir.joinpath(Path("out"))
    os.mkdir(out_dir)

    if save_strategy:
        wd = Watchdog(out_dir, params["NumberOfResolutions"])
        wd.set_strategy(save_strategy)
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

    if save_strategy:
        wd.stop()
        wd.join()
        wd.sv_strategy.close()

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
        Parameters.from_base(mesh_size=3, sampling_p=0.03, write_img=True)
        .multi_resolution(1, p_sched=[5, 5, 5])
        .asgd()
        .instance(Collection.EMPIRE, 16)
        .stopping_criteria(iterations=50)
    )
    run(params, Path("output/" + str(params)))
