from pathlib import Path
import os
import time
import subprocess
import logging
from typing import Any, Dict, List

import pandas as pd
import numpy as np

from thesispy.elastix_wrapper import TimeoutException, time_limit
from thesispy.elastix_wrapper.parameters import Parameters
from thesispy.elastix_wrapper.watchdog import SaveStrategy, Watchdog
from thesispy.experiments.validation import calc_validation
from thesispy.experiments.instance import (
    get_instance,
    get_np_array,
    read_deformed_lms,
    RunResult,
    read_controlpoints,
    read_transform_params,
)
from thesispy.definitions import *

ELASTIX = os.environ.get("ELASTIX_EXECUTABLE")
TRANSFORMIX = os.environ.get("TRANSFORMIX_EXECUTABLE")
logger = logging.getLogger("Wrapper")


def run(
    params_list: List[Parameters],
    run_dir: Path,
    save_strategy: SaveStrategy = None,
    suppress_stdout: bool = True,
    visualize: bool = False,
    validate: bool = True,
) -> Dict[str, Any]:
    time_start = time.perf_counter()

    if type(params_list) is not list:
        params_list = [params_list]

    run_dir.mkdir(parents=True)
    param_files = []
    main_params = params_list[-1]
    for i, params in enumerate(params_list):
        param_files.append(params.write(run_dir, i + 1))

    out_dir = run_dir.joinpath(Path("out"))
    if save_strategy:
        wd = Watchdog(out_dir, main_params["NumberOfResolutions"])
        wd.set_strategy(save_strategy)
        wd.start()

    finished = False
    run_result = None
    logger.info(f"Running elastix in: {str(run_dir)}")
    try:
        execute_elastix(param_files, out_dir, main_params, suppress_stdout)
        finished = True
    except TimeoutException:
        logger.warning(f"Exceeded time limit of {main_params['MaxTimeSeconds']} seconds.")
    except KeyboardInterrupt:
        logger.warning(f"Run ended prematurely by user.")
    except Exception as e:
        logger.error(f"Run ended with exception: {e}")
    finally:
        if finished and validate:
            val_metrics, run_result = validation(main_params, run_dir)

        if save_strategy:
            if finished and validate:
                for metric in val_metrics:
                    wd.sv_strategy.save_custom(metric)
            wd.stop()
            wd.join()
            wd.sv_strategy.close(finished)

        time_end = time.perf_counter()
        logger.info(f"Run ended. It took {time_end - time_start:0.4f} seconds")

        if finished and visualize:
            execute_visualize(out_dir)

    return run_result


def execute_elastix(
    param_files: List[Path], out_dir: Path, params: Parameters, suppress_stdout: bool = True
):
    param_files_args = [["-p", str(param_file)] for param_file in param_files]
    param_files_args = [item for sublist in param_files_args for item in sublist]

    with time_limit(params["MaxTimeSeconds"]):
        args = [
            ELASTIX,
            *param_files_args,
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
        env = None
        if params["Optimizer"] == "AdaptiveStochasticGradientDescent":
            env = os.environ.copy()
            env["OMP_WAIT_POLICY"] = "PASSIVE"

        subprocess.run(args, check=True, stdout=output, env=env)


def generate_transformed_points(
    params_file: Path, out_dir: Path, points_file: Path = None, moving_img_path: Path = None
):
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
    if moving_img_path is not None:
        args += ["-in", str(moving_img_path.resolve())]
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def execute_visualize(out_dir: Path):
    visualizers = ["vv", "mitk", "Slicer"]
    visualizer = None
    for vis in visualizers:
        if not subprocess.run(f"command -v {vis}", shell=True).returncode:
            visualizer = vis
            break

    if visualizer:
        subprocess.run(
            [visualizer, str((out_dir / "result.0.mhd").resolve())], cwd=str(out_dir.resolve())
        )


def get_run_result(collection: Collection, instance_id: int, transform_params: Path):
    out_dir = transform_params.parent.resolve()
    instance = get_instance(collection, instance_id)
    run_result = RunResult(instance)
    if instance.lms_fixed is not None:
        generate_transformed_points(transform_params, out_dir, instance.lms_fixed_path)
        run_result.deformed_lms = read_deformed_lms(out_dir / "outputpoints.txt")

    if instance.surface_points_paths is not None:
        run_result.deformed_surface_points = []
        for surface_points_path in instance.surface_points_paths:
            generate_transformed_points(transform_params, out_dir, surface_points_path)
            run_result.deformed_surface_points.append(
                read_deformed_lms(out_dir / "outputpoints.txt")
            )

    generate_transformed_points(transform_params, out_dir, moving_img_path=instance.moving_path)
    run_result.dvf = get_np_array(out_dir / "deformationField.mhd")
    run_result.deformed = get_np_array(out_dir / "result.mhd")
    run_result.control_points = read_controlpoints(out_dir / "controlpoints.dat")
    _, spacing, origin = read_transform_params(transform_params)
    run_result.grid_spacing = spacing
    run_result.grid_origin = origin
    final_evals = pd.read_csv(out_dir / "final_evals.txt", sep=",", index_col=0, header=None)
    nr_voxels = (
        np.sum(run_result.instance.mask)
        if run_result.instance.mask is not None
        else np.prod(run_result.deformed.shape)
    )
    run_result.bending_energy = final_evals.loc["bending_energy"].values[0] * nr_voxels

    return run_result


def validation(params: Parameters, run_dir: Path):
    out_dir = run_dir.joinpath(Path("out"))
    transform_params = out_dir / "TransformParameters.0.txt"
    run_result = get_run_result(
        Collection(params["Collection"]), int(params["Instance"]), transform_params
    )

    return calc_validation(run_result), run_result


if __name__ == "__main__":
    params_main = (
        Parameters.from_base(
            mesh_size=7, use_mask=True, metric="AdvancedNormalizedCorrelation", seed=88
        )
        .asgd()
        .regularize(0.01)
        .multi_resolution(3, p_sched=[4, 3, 2])
        .stopping_criteria(iterations=[200, 200, 400])
        .instance(Collection.LEARN, 1)
    )
    run(
        [params_main],
        Path("output/" + str(params_main)),
        suppress_stdout=False,
        visualize=True,
        validate=True,
    )
