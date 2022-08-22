from pathlib import Path
import os
import time
import subprocess
import logging

from typing import Any, Dict

import numpy as np
import nibabel as nib

from thesispy.elastix_wrapper import TimeoutException, time_limit
from thesispy.elastix_wrapper.parameters import Collection, GOMEAType, Parameters
from thesispy.elastix_wrapper.watchdog import SaveStrategy, SaveStrategyPrint, Watchdog

ELASTIX = os.environ.get("ELASTIX_EXECUTABLE")
TRANSFORMIX = os.environ.get("TRANSFORMIX_EXECUTABLE")
logger = logging.getLogger("Wrapper")


def run(
    params: Parameters,
    run_dir: Path,
    save_strategy: SaveStrategy = None,
    suppress_stdout: bool = True,
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
        params.compute_tre = False
    except KeyboardInterrupt:
        logger.info(f"Run ended prematurely by user.")
        params.compute_tre = False

    if save_strategy:
        if params.compute_tre:
            tre = compute_tre(
                out_dir,
                params.lms_fixed_path,
                params.lms_moving_path,
                params.fixed_path,
                params["Collection"]
            )
            logger.info(f"TRE: {tre}")
            wd.sv_strategy.save_custom("TRE", tre)
        wd.stop()
        wd.join()
        wd.sv_strategy.close()

    time_end = time.perf_counter()

    logger.info(f"Run ended successfully. It took {time_end - time_start:0.4f} seconds")

    if params["WriteResultImage"]:
        execute_visualize(out_dir)


def compute_tre(out_dir: Path, lms_fixed: Path, lms_moving: Path, img_fixed: Path, collection: Collection):
    params_file = out_dir / "TransformParameters.0.txt"
    lms_moving = np.loadtxt(lms_moving, skiprows=2)

    spacing = np.ones(3)
    if collection == Collection.LEARN:
        image = nib.load(img_fixed)
        spacing = np.array(image.header.get_zooms())

    try:
        execute_transformix(params_file, lms_fixed, out_dir)
    except subprocess.CalledProcessError as err:
        err_msg = err.stderr.decode("utf-8").strip("\n")
        logger.error(
            f"Something went wrong while running transformix at: {str(out_dir)}, {err_msg}"
        )
        return

    warped_points = out_dir / "outputpoints.txt"
    lms_fixed_warped = np.zeros(lms_moving.shape)
    with open(warped_points) as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            s = line.split(";")[3]
            s = s[s.find("[") + 1 : s.find("]")].split(" ")
            index = np.array([float(s[1]), float(s[2]), float(s[3])])
            lms_fixed_warped[i] = index

    return np.linalg.norm((lms_fixed_warped - lms_moving) * spacing, axis=1).mean()


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
        if params.compute_tre:
            args += [
                "-fp",
                str(params.lms_fixed_path.resolve()),
                "-mp",
                str(params.lms_moving_path.resolve()),
            ]

        output = subprocess.DEVNULL if suppress_stdout else None
        env = os.environ.copy()
        if params["Optimizer"] == "AdaptiveStochasticGradientDescent":
            env["OMP_WAIT_POLICY"] = "PASSIVE"

        subprocess.run(args, check=True, stdout=output, stderr=subprocess.PIPE, env=env)


def execute_transformix(params_file: Path, points_file: Path, out_dir: Path):
    args = [
        TRANSFORMIX,
        "-tp",
        str(params_file),
        "-def",
        str(points_file),
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
        Parameters.from_base(mesh_size=3, metric="AdvancedMeanSquares")
        .asgd()
        .result_image()
        .regularize(0.001, False)
        .stopping_criteria(200)
        .instance(Collection.SYNTHETIC, 2)
    )
    run(params, Path("output/" + str(params)), SaveStrategy(), False)
