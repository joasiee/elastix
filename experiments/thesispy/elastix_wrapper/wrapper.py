from pathlib import Path
import os
import subprocess
import logging

from typing import Any, Dict

import numpy as np
import nibabel as nib

from thesispy.elastix_wrapper import TimeoutException, time_limit
from thesispy.elastix_wrapper.parameters import Collection, Parameters
from thesispy.elastix_wrapper.watchdog import SaveStrategy, Watchdog

ELASTIX = os.environ.get("ELASTIX_EXECUTABLE")
TRANSFORMIX = os.environ.get("TRANSFORMIX_EXECUTABLE")
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
        err_msg = err.stderr.decode('utf-8').strip('\n')
        logger.error(
            f"Something went wrong while running elastix at: {str(run_dir)}: {err_msg}"
        )
    except TimeoutException:
        logger.info(f"Exceeded time limit of {params['MaxTimeSeconds']} seconds.")
    except KeyboardInterrupt:
        logger.info(f"Run ended prematurely by user.")

    if save_strategy:
        if params.compute_tre:
            tre = compute_tre(out_dir, params.lms_fixed_path, params.lms_moving_path, params.fixed_path)
            logger.info(f"TRE: {tre}")
            wd.sv_strategy.save_custom("TRE", tre)
        wd.stop()
        wd.join()
        wd.sv_strategy.close()

    logger.info("Run ended successfully.")


def compute_tre(out_dir: Path, lms_fixed: Path, lms_moving: Path, img_fixed: Path):
    params_file = out_dir / "TransformParameters.0.txt"
    points_file = out_dir / "lms_points.txt"
    lms_fixed = np.loadtxt(lms_fixed)
    lms_moving = np.loadtxt(lms_moving)
    image = nib.load(img_fixed)
    spacing = np.array(image.header.get_zooms())

    with open(points_file, "wb") as file:
        n_points = len(lms_fixed)
        file.write(b"index\n")
        file.write(bytes(str(n_points), encoding="utf-8") + b"\n")
        np.savetxt(file, lms_fixed, fmt='%f')

    try:
        execute_transformix(params_file, points_file, out_dir)
    except subprocess.CalledProcessError as err:
        err_msg = err.stderr.decode('utf-8').strip('\n')
        logger.error(
            f"Something went wrong while running transformix at: {str(out_dir)}, {err_msg}"
        )
        return

    warped_points = out_dir / "outputpoints.txt"
    lms_fixed_warped = np.zeros(lms_fixed.shape)
    with open(warped_points) as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            s = line.split(";")[3]
            s = s[s.find("[")+1:s.find("]")].split(" ")
            index = np.array([float(s[1]), float(s[2]), float(s[3])])
            lms_fixed_warped[i] = index

    return np.linalg.norm((lms_fixed_warped - lms_moving) * spacing, axis=1).mean()

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

        subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


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


if __name__ == "__main__":
    params = (
        Parameters.from_base(mesh_size=5, sampler="Full", seed=1)
        .multi_resolution(1, [4, 4, 4])
        .asgd()
        .instance(Collection.LEARN, 1)
        .stopping_criteria(iterations=[3000])
    )
    run(params, Path("output/" + str(params)), SaveStrategy())
