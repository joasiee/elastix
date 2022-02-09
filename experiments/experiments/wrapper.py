from pathlib import Path
import os
import tempfile
import subprocess
import logging
from timeit import default_timer as timer
from typing import Any, Dict
import pandas as pd
from experiments import TimeoutException, time_limit
from bson.binary import Binary

from experiments.parameters import Collection, Parameters

ELASTIX = os.environ.get("ELASTIX_EXECUTABLE")
logger = logging.getLogger("Wrapper")
app_logger = logging.getLogger("AppOutput")


def run(params: Parameters) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        params_file = params.write(Path(tmp_dir))
        out_dir = Path(tmp_dir).joinpath(Path("out"))
        os.mkdir(out_dir)

        logger.info(f"Starting elastix for: {str(params)}.")
        start = timer()
        try:
            execute_elastix(params_file, out_dir, params)
        except subprocess.CalledProcessError as err:
            logger.error(
                f"Something went wrong while running elastix with params: {str(params)}: {err.stderr}")
            return
        except TimeoutException:
            logger.info(
                f"Exceeded time limit of {params['MaxTimeSeconds']} seconds.")
        except KeyboardInterrupt:
            logger.info(f"Run ended prematurely by user, saving results.")

        logger.info("Run finished successfully.")
        results = get_output(out_dir, params, timer() - start)
        return results


def get_output(out_dir: Path, params: Parameters, duration: float):
    results = {
        "id": str(params),
        "collection": params["Collection"],
        "instance": params["Instance"],
        "params": params.params,
        "resolutions": [],
        "duration": duration
    }
    for r in range(params["NumberOfResolutions"]):
        try:
            resolution_results = pd.read_csv(
                out_dir / f"IterationInfo.0.R{r}.txt", sep="	").to_dict()
            results["resolutions"].append(
                {k: list(v.values()) for k, v in resolution_results.items()})
        except FileNotFoundError:
            logger.warning(f"Output file for resolution {r} not found.")

    results["final_metric"] = results["resolutions"][-1]["2:Metric"][-1]
    i = 0
    while f"Metric{i}Weight" in params.params:
        results[f"final_metric_{i}"] = results["resolutions"][-1][f"2:Metric{i}"][-1]
        i += 1

    return results


def execute_elastix(params_file: Path, out_dir: Path, params: Parameters):
    with open("app.log", "w") as out, time_limit(params["MaxTimeSeconds"]):
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
            check=True,
            stdout=out,
            stderr=out
        )


if __name__ == "__main__":
    params = (
        Parameters(sampler="Random", mesh_size=5)
        .gomea()
        .instance(Collection.EMPIRE, 1)
        .stopping_criteria(iterations=2)
    )
    print(run(params))
