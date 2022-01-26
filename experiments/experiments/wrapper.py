from pathlib import Path
import os
import tempfile
import subprocess
import logging
from timeit import default_timer as timer
import pandas as pd
from experiments import TimeoutException, time_limit

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
            try:
                Wrapper.execute_elastix(params_file, out_dir, params)
            except subprocess.CalledProcessError as err:
                logger.error(
                    f"Something went wrong while running elastix with params: {str(params)}: {err.stderr}")
            except TimeoutException:
                logger.info(f"Exceeded time limit of {params['MaxTimeSeconds']} seconds.")
                pass

            logger.info("Run finished successfully.")
            self.save_output(out_dir, params, timer() - start)

    def save_output(self, out_dir: Path, params: Parameters, duration: float):
        results = {
            "id": str(params),
            "params": params.params,
            "resolutions": [],
            "duration": duration
        }
        for r in range(params["NumberOfResolutions"]):
            resolution_results = pd.read_csv(
                out_dir / f"IterationInfo.0.R{r}.txt", sep="	").to_dict()
            results["resolutions"].append(
                {k: list(v.values()) for k, v in resolution_results.items()})

        results["final_metric"] = results["resolutions"][-1]["2:Metric"][-1]

        self.db.save_results(results)

    @staticmethod
    def execute_elastix(params_file: Path, out_dir: Path, params: Parameters):
        with open("app.log", "wb") as out, time_limit(params["MaxTimeSeconds"]):
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
        Parameters(sampler="Full", mesh_size=10)
        .gomea()
        .instance(Collection.EXAMPLES, 1)
        .stopping_criteria(iterations=200, max_time_s=10)
    )
    wrap = Wrapper()
    wrap.run(params)
