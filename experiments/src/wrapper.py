from pathlib import Path
import os
import tempfile
import subprocess
from typing import List

from src.parameters import Collection, Parameters
from src.db import DBClient

ELASTIX = os.environ.get("ELASTIX_EXECUTABLE")


class Wrapper:
    def __init__(self) -> None:
        self.db = DBClient()

    def run(self, params: Parameters):
        with tempfile.TemporaryDirectory() as tmp_dir:
            params_file = params.write(Path(tmp_dir))
            out_dir = Path(tmp_dir).joinpath(Path("out"))
            os.mkdir(out_dir)

            result = subprocess.run(
                [
                    ELASTIX,
                    "-p",
                    str(params_file),
                    "-f",
                    str(params.fixed_path),
                    "-m",
                    str(params.moving_path),
                    "-out",
                    str(out_dir),
                ]
            )
            if result.returncode == 0:
                self.get_output(out_dir)

    def get_output(self, out_dir: Path):
        pass

    def get_output_resolution(r: int):
        pass


if __name__ == "__main__":
    params = (
        Parameters(sampler="Full")
        .gomea(iterations=20, pop_size=40)
        .instance(Collection.EMPIRE, 1)
    )
    wrap = Wrapper()
    wrap.run(params)
