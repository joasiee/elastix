from pathlib import Path
import os
import tempfile
import subprocess
import logging
from timeit import default_timer as timer
import pandas as pd
from experiments import TimeoutException, time_limit
from bson.binary import Binary

from experiments.parameters import Collection, Parameters
from experiments.db import DBClient

ELASTIX = os.environ.get("ELASTIX_EXECUTABLE")
logger = logging.getLogger("Wrapper")
app_logger = logging.getLogger("AppOutput")


class Wrapper:
    def __init__(self) -> None:
        self.db = DBClient()

    def run(self, params: Parameters):
        with tempfile.TemporaryDirectory() as tmp_dir:
            params_file = params.write(Path(tmp_dir))
            out_dir = Path(tmp_dir).joinpath(Path("out"))
            os.mkdir(out_dir)

            logger.info(f"Starting elastix for: {str(params)}.")
            start = timer()
            app_log = None
            try:
                app_log = Wrapper.execute_elastix(params_file, out_dir, params)
            except subprocess.CalledProcessError as err:
                logger.error(
                    f"Something went wrong while running elastix with params: {str(params)}: {err.stderr}")
                return
            except TimeoutException:
                logger.info(
                    f"Exceeded time limit of {params['MaxTimeSeconds']} seconds.")
                pass

            logger.info("Run finished successfully.")
            return self.save_output(out_dir, params, timer() - start, app_log)

    def save_output(self, out_dir: Path, params: Parameters, duration: float, app_log: Binary):
        results = {
            "id": str(params),
            "collection": params["Collection"],
            "instance": params["Instance"],
            "params": params.params,
            "resolutions": [],
            "duration": duration,
            "log": app_log
        }
        for r in range(params["NumberOfResolutions"]):
            resolution_results = pd.read_csv(
                out_dir / f"IterationInfo.0.R{r}.txt", sep="	").to_dict()
            results["resolutions"].append(
                {k: list(v.values()) for k, v in resolution_results.items()})

        results["final_metric"] = results["resolutions"][-1]["2:Metric"][-1]
        i = 0
        while f"Metric{i}Weight" in params.params:
            results[f"final_metric_{i}"] = results["resolutions"][-1][f"2:Metric{i}"][-1]
            i += 1

        self.db.save_results(results)
        return results

    @staticmethod
    def execute_elastix(params_file: Path, out_dir: Path, params: Parameters) -> str:
        with open("app.log", "w+") as out, time_limit(params["MaxTimeSeconds"]):
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
            out.seek(0)
            return out.read()


if __name__ == "__main__":
    params = (
        Parameters(sampler="Random", mesh_size=8)
        .gomea()
        .instance(Collection.EMPIRE, 1)
        .stopping_criteria(iterations=1)
        .multi_metric()
    )
    wrap = Wrapper()
    wrap.run(params)
