from pathlib import Path
import os
import subprocess
import logging
from typing import List

import pandas as pd

from thesispy.elastix_wrapper.parameters import Parameters
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


def execute_elastix(param_files: List[Path], out_dir: Path, params: Parameters, suppress_stdout: bool = True):
    param_files_args = [["-p", str(param_file)] for param_file in param_files]
    param_files_args = [item for sublist in param_files_args for item in sublist]

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
    """Use transformix to generate transformed landmarks given a transform definition.

    If no landmarks are provided, all voxels are transformed and a deformation field is generated.
    """
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
    result = subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return result.returncode


def transformix_image(params_file: Path, out_dir: Path, moving_img_path: Path, interpolator: str = None):
    """Use transformix to transform an image given a transform definition."""
    curr_interp = read_key_from_transform_params(params_file, "ResampleInterpolator")
    if interpolator is not None:
        change_key_in_transform_params(params_file, "ResampleInterpolator", interpolator)

    args = [
        TRANSFORMIX,
        "-tp",
        str(params_file),
        "-out",
        str(out_dir),
        "-threads",
        os.environ["OMP_NUM_THREADS"],
        "-in",
        str(moving_img_path.resolve()),
    ]
    result = subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    change_key_in_transform_params(params_file, "ResampleInterpolator", curr_interp)

    return result.returncode


def read_key_from_transform_params(transform_params: Path, key: str):
    with open(transform_params, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("(" + key):
            return line.split()[1].strip(")")
    return None


def change_key_in_transform_params(transform_params: Path, key: str, value: str):
    with open(transform_params, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("(" + key):
            lines[i] = f"({key} {value})\n"
            break
    with open(transform_params, "w") as f:
        f.writelines(lines)


def execute_visualize(out_dir: Path):
    """Visualize registration result using the first available visualizer from [vv, mitk, slicer]."""
    visualizers = ["vv", "mitk", "Slicer"]
    visualizer = None
    for vis in visualizers:
        if not subprocess.run(f"command -v {vis}", shell=True).returncode:
            visualizer = vis
            break

    if visualizer:
        subprocess.run([visualizer, str((out_dir / "result.0.mhd").resolve())], cwd=str(out_dir.resolve()))


def get_run_result(collection: Collection, instance_id: int, transform_params: Path):
    """Given a collection, instance and transform parameters, return the run result.

    Using the transform, the landmarks are deformed, the moving image is transformed, 
    and all other required data for validation is stored in a RunResult object.
    """
    out_dir = transform_params.parent.resolve()
    out_dir_transform = out_dir / "transform"
    out_dir_transform.mkdir(parents=True, exist_ok=True)

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

    if instance.moving_mask_path is not None:
        transformix_image(
            transform_params, out_dir_transform, instance.moving_mask_path, "FinalNearestNeighborInterpolator"
        )
        run_result.deformed_mask = get_np_array(out_dir_transform / "result.mhd")
        if collection == Collection.LEARN:
            run_result.deformed_mask[run_result.instance.fixed == -1024] = 0

    run_result.dvf = get_np_array(out_dir / "deformationField.mhd")
    run_result.deformed = get_np_array(out_dir / "result.mhd")
    run_result.control_points = read_controlpoints(out_dir / "controlpoints.dat")
    _, spacing, origin = read_transform_params(transform_params)
    run_result.grid_spacing = spacing
    run_result.grid_origin = origin

    if (out_dir / "final_evals.txt").exists():
        final_evals = pd.read_csv(out_dir / "final_evals.txt", sep=",", index_col=0, header=None)
        run_result.bending_energy = final_evals.loc["bending_energy"].values[0]

    run_result.transform_params = transform_params.absolute().resolve()

    return run_result
