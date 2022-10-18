from pathlib import Path
import os
import time
import subprocess
import logging

from typing import Any, Dict

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
from thesispy.definitions import Collection, LinkageType

ELASTIX = os.environ.get("ELASTIX_EXECUTABLE")
TRANSFORMIX = os.environ.get("TRANSFORMIX_EXECUTABLE")
logger = logging.getLogger("Wrapper")


def run(
    params: Parameters,
    run_dir: Path,
    save_strategy: SaveStrategy = None,
    suppress_stdout: bool = True,
    visualize: bool = False,
    validate: bool = True,
) -> Dict[str, Any]:
    time_start = time.perf_counter()

    run_dir.mkdir(parents=True)
    params_file = params.write(run_dir)
    out_dir = run_dir.joinpath(Path("out"))
    if save_strategy:
        wd = Watchdog(out_dir, params["NumberOfResolutions"])
        wd.set_strategy(save_strategy)
        wd.start()

    logger.info(f"Running elastix in: {str(run_dir)}")
    try:
        execute_elastix(params_file, out_dir, params, suppress_stdout)
    except subprocess.CalledProcessError as err:
        err_msg = err.stderr.decode("utf-8").strip("\n")
        logger.error(f"Something went wrong while running elastix at: {str(run_dir)}: {err_msg}")
    except TimeoutException:
        logger.info(f"Exceeded time limit of {params['MaxTimeSeconds']} seconds.")
    except KeyboardInterrupt:
        logger.info(f"Run ended prematurely by user.")

    run_result = None
    if validate:
        val_metrics, run_result = validation(params, run_dir)

    if save_strategy:
        if validate:
            for metric in val_metrics:
                wd.sv_strategy.save_custom(metric)
        wd.stop()
        wd.join()
        wd.sv_strategy.close()

    time_end = time.perf_counter()
    logger.info(f"Run ended successfully. It took {time_end - time_start:0.4f} seconds")

    if visualize:
        execute_visualize(out_dir)

    return run_result


def execute_elastix(params_file: Path, out_dir: Path, params: Parameters, suppress_stdout: bool = True):
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
        env = None
        if params["Optimizer"] == "AdaptiveStochasticGradientDescent":
            env = os.environ.copy()
            env["OMP_WAIT_POLICY"] = "PASSIVE"

        subprocess.run(args, check=True, stdout=output, stderr=subprocess.PIPE, env=env)


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
        subprocess.run([visualizer, str((out_dir / "result.0.mhd").resolve())], cwd=str(out_dir.resolve()))


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
            run_result.deformed_surface_points.append(read_deformed_lms(out_dir / "outputpoints.txt"))

    generate_transformed_points(transform_params, out_dir, moving_img_path=instance.moving_path)
    run_result.dvf = get_np_array(out_dir / "deformationField.mhd")
    run_result.deformed = get_np_array(out_dir / "result.mhd")
    run_result.control_points = read_controlpoints(out_dir / "controlpoints.dat")
    _, spacing, origin = read_transform_params(transform_params)
    run_result.grid_spacing = spacing
    run_result.grid_origin = origin

    return run_result


def validation(params: Parameters, run_dir: Path):
    out_dir = run_dir.joinpath(Path("out"))
    transform_params = out_dir / "TransformParameters.0.txt"
    run_result = get_run_result(Collection(params["Collection"]), int(params["Instance"]), transform_params)

    return calc_validation(run_result), run_result

if __name__ == "__main__":
    params_main = (
        Parameters.from_base(mesh_size=5, metric="AdvancedMeanSquares", seed=2, use_mask=True)
        .gomea(LinkageType.STATIC_EUCLIDEAN, max_set_size=3)
        .stopping_criteria(10)
        .instance(Collection.SYNTHETIC, 1)
    )
    run(params_main, Path("output/" + str(params_main)), suppress_stdout=False, visualize=True)
