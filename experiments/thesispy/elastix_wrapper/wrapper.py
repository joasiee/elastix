from pathlib import Path
import os
import time
import subprocess
import logging

from typing import Any, Dict

import numpy as np
import nibabel as nib

from thesispy.elastix_wrapper import TimeoutException, time_limit
from thesispy.elastix_wrapper.parameters import Collection, Parameters
from thesispy.elastix_wrapper.watchdog import SaveStrategy, Watchdog
from thesispy.experiments.instance import get_np_array, read_deformed_lms, get_instance
from thesispy.experiments.validation import calc_validation

ELASTIX = os.environ.get("ELASTIX_EXECUTABLE")
TRANSFORMIX = os.environ.get("TRANSFORMIX_EXECUTABLE")
logger = logging.getLogger("Wrapper")


def run(
    params: Parameters,
    run_dir: Path,
    save_strategy: SaveStrategy = None,
    suppress_stdout: bool = True,
    visualize: bool = False,
) -> Dict[str, Any]:
    time_start = time.perf_counter()

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
        execute_elastix(params_file, out_dir, params, suppress_stdout)
    except subprocess.CalledProcessError as err:
        err_msg = err.stderr.decode("utf-8").strip("\n")
        logger.error(
            f"Something went wrong while running elastix at: {str(run_dir)}: {err_msg}"
        )
    except TimeoutException:
        logger.info(f"Exceeded time limit of {params['MaxTimeSeconds']} seconds.")
    except KeyboardInterrupt:
        logger.info(f"Run ended prematurely by user.")

    val_metrics = validation(params, run_dir)
    val_metrics = {"Validation/" + str(key): val for key, val in val_metrics.items()}

    if save_strategy:
        wd.sv_strategy.save_custom(val_metrics)
        wd.stop()
        wd.join()
        wd.sv_strategy.close()

    time_end = time.perf_counter()
    logger.info(f"Run ended successfully. It took {time_end - time_start:0.4f} seconds")

    if visualize and not save_strategy:
        execute_visualize(out_dir)

def validation(params: Parameters, run_dir: Path):
    out_dir = run_dir.joinpath(Path("out"))
    transform_params = out_dir / "TransformParameters.0.txt"
    instance = get_instance(Collection(params['Collection']), int(params['Instance']))
    deformed, dvf, deformed_lms = None, None, None

    if instance.lms_fixed:
        generate_transformed_points(transform_params, params.lms_fixed_path, out_dir)
        deformed_lms = read_deformed_lms(out_dir / "outputpoints.txt")
    
    generate_transformed_points(transform_params, None, out_dir)
    dvf = get_np_array(out_dir / "deformationField.mhd")

    deformed = get_np_array(out_dir / "result.0.mhd")

    return calc_validation(instance, deformed, dvf, deformed_lms)


def execute_elastix(
    params_file: Path, out_dir: Path, params: Parameters, suppress_stdout: bool = True
):
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

        output = subprocess.DEVNULL if suppress_stdout else None
        env = os.environ.copy()
        env["OMP_WAIT_POLICY"] = "PASSIVE"

        subprocess.run(args, check=True, stdout=output, stderr=subprocess.PIPE, env=env)


def generate_transformed_points(params_file: Path, points_file: Path, out_dir: Path):
    args = [
        TRANSFORMIX,
        "-tp",
        str(params_file),
        "-def",
        str(points_file) if points_file else "all",
        "-out",
        str(out_dir),
        "-threads",
        os.environ["OMP_NUM_THREADS"],
    ]
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def execute_visualize(out_dir: Path):
    visualizers = ["vv", "mitk", "slicer"]
    visualizer = None
    for vis in visualizers:
        if not subprocess.run(["command", "-v", vis], shell=True).returncode:
            visualizer = vis
            break
    
    if visualizer:
        subprocess.run(
            [visualizer, str((out_dir / "result.0.mhd").resolve())],
            cwd=str(out_dir.resolve())
        )


if __name__ == "__main__":
    params = (
        Parameters.from_base(mesh_size=8, metric="AdvancedMeanSquares", seed=1, use_mask=False)
        .asgd()
        .result_image()
        # .regularize(1e-9, True)
        .stopping_criteria(5)
        .instance(Collection.SYNTHETIC, 1)
    )
    run(params, Path("output/" + str(params)), SaveStrategy(), False, True)
