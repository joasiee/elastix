from __future__ import annotations
from datetime import datetime

import json
import logging
from math import ceil
import os
import time
from PIL import Image
import nibabel as nib
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List
from thesispy.definitions import ROOT_DIR

BASE_PARAMS_PATH = ROOT_DIR / Path("resources", "base_params.json")
INSTANCE_CONFIG_PATH = ROOT_DIR / Path("resources", "instances.json")
INSTANCES_CONFIG: Dict[str, str] = {}
INSTANCES_SRC = Path(os.environ.get("INSTANCES_SRC"))

with INSTANCE_CONFIG_PATH.open() as f:
    INSTANCES_CONFIG = json.loads(f.read())

logger = logging.getLogger("Parameters")


class Collection(str, Enum):
    EMPIRE = "EMPIRE"
    LEARN = "LEARN"
    EXAMPLES = "EXAMPLES"
    SYNTHETIC = "SYNTHETIC"


class GOMEAType(Enum):
    GOMEA_FULL = -1
    GOMEA_UNIVARIATE = 1
    GOMEA_CP = -6


class Parameters:
    def __init__(self, params) -> None:
        self.params = params
        self.id = [str(int(time.time())), str(datetime.now().microsecond)]

    @classmethod
    def from_base(
        cls,
        metric: str = "AdvancedMeanSquares",
        mesh_size: List[int] | int = 12,
        seed: int = None,
    ):
        with BASE_PARAMS_PATH.open() as f:
            params: Dict[str, Any] = json.loads(f.read())
        return cls(params).args(
            {"Metric": metric, "MeshSize": mesh_size, "RandomSeed": seed}
        )

    @classmethod
    def from_json(cls, jsondump):
        params = json.loads(jsondump)
        return cls(params).set_paths()

    def instance(self, collection: Collection, instance: int) -> Parameters:
        self["Collection"] = collection.value
        self["Instance"] = instance
        self.set_paths()
        return self.calc_voxel_params()

    def multi_metric(
        self,
        metric0: str = "AdvancedMeanSquares",
        metric1: str = "TransformBendingEnergyPenaltyAnalytic",
        weight0: float = 1.0,
        weight1: float = 0.01,
        params: Dict[str, Any] = None,
    ) -> Parameters:
        self.n_param("FixedImagePyramid", 2)
        self.n_param("MovingImagePyramid", 2)
        self.n_param("Interpolator", 2)
        self.n_param("ImageSampler", 2)
        return self.args(
            {
                "Registration": "MultiMetricMultiResolutionRegistration",
                "Metric": [metric0, metric1],
                "Metric0Weight": weight0,
                "Metric1Weight": weight1,
            },
            params,
        )

    def multi_resolution(
        self, n: int = 3, p_sched: List[int] = None, g_sched: List[int] = None
    ) -> Parameters:
        return self.args(
            {
                "NumberOfResolutions": n,
                "ImagePyramidSchedule": p_sched,
                "GridSpacingSchedule": g_sched,
            }
        )

    def stopping_criteria(
        self,
        iterations: List[int] | int = None,
        evals: List[int] | int = None,
        max_time_s: int = 0,
        fitness_var: float = 1e-9,
    ):
        return self.args(
            {
                "MaximumNumberOfIterations": iterations,
                "MaxNumberOfEvaluations": evals,
                "MaxTimeSeconds": max_time_s,
                "FitnessVarianceTolerance": fitness_var,
            }
        )

    def gomea(
        self,
        fos: GOMEAType = GOMEAType.GOMEA_FULL,
        pop_size: List[int] | int = None,
        shrinkage: bool = False,
    ) -> Parameters:
        pevals = False if fos == GOMEAType.GOMEA_FULL else True
        return self.args(
            {
                "Optimizer": "GOMEA",
                "OptimizerName": fos.name,
                "FosElementSize": fos.value,
                "BasePopulationSize": pop_size,
                "PartialEvaluations": pevals,
                "UseShrinkage": shrinkage,
            }
        )

    def asgd(self, params: Dict[str, Any] = None):
        return self.args({"Optimizer": "AdaptiveStochasticGradientDescent"}, params)

    def debug(self):
        return self.args(
            {
                "WritePyramidImagesAfterEachResolution": True,
                "WriteSamplesEveryIteration": True,
                "WriteMeanPointsEveryIteration": True,
                "WriteResultImage": True,
            }
        )

    def sampler(self, sampler, pct: float = 0.05):
        return self.args({"ImageSampler": sampler, "SamplingPercentage": pct})

    def args(
        self, params: Dict[str, Any], extra_params: Dict[str, Any] = None
    ) -> Parameters:
        for key, value in params.items():
            self[key] = value
        if extra_params is not None:
            for key, value in extra_params.items():
                self[key] = value
        return self

    def prune(self):
        self.params = {k: v for k, v in self.params.items() if v is not None}

    def write(self, dir: Path) -> None:
        self.prune()
        out_file = dir.joinpath(Path("params.txt"))
        with open(str(out_file), "w+") as f:
            for key, value in self.params.items():
                f.write(Parameters.parse_param(key, value))
        return out_file

    def calc_voxel_params(self) -> Parameters:
        voxel_dims = self.get_voxel_dimensions()

        if not isinstance(self["MeshSize"], List):
            self["MeshSize"] = [self["MeshSize"] for _ in range(len(voxel_dims))]

        if not self["GridSpacingSchedule"]:
            self["GridSpacingSchedule"] = [
                1 for _ in range(self["NumberOfResolutions"])
            ]

        if not self["ImagePyramidSchedule"]:
            self["ImagePyramidSchedule"] = [
                2**n
                for n in range(self["NumberOfResolutions"] - 1, -1, -1)
                for _ in range(len(voxel_dims))
            ]

        elif len(self["ImagePyramidSchedule"]) == self["NumberOfResolutions"]:
            self["ImagePyramidSchedule"] = [
                self["ImagePyramidSchedule"][i]
                for i in range(self["NumberOfResolutions"])
                for _ in range(len(voxel_dims))
            ]

        voxel_spacings = []
        total_samples = [1] * self["NumberOfResolutions"]
        for i, voxel_dim in enumerate(voxel_dims):
            voxel_spacings.append(ceil(voxel_dim / self["MeshSize"][i]))
            for n in range(len(total_samples)):
                total_samples[n] *= int(
                    voxel_dim / self["ImagePyramidSchedule"][n * len(voxel_dims) + i]
                )

        self["FinalGridSpacingInVoxels"] = voxel_spacings

        self["NumberOfSpatialSamples"] = total_samples
        if self["ImageSampler"] != "Full":
            self["NumberOfSpatialSamples"] = [
                int(x * self["SamplingPercentage"])
                for x in self["NumberOfSpatialSamples"]
            ]

        return self

    def get_voxel_dimensions(self) -> List[int]:
        extension = INSTANCES_CONFIG[self["Collection"]]["extension"]
        if extension == "mhd":
            return Parameters.read_mhd(self.fixed_path)["DimSize"]
        elif extension == "png":
            image = Image.open(str(self.fixed_path))
            return list(image.size)
        elif extension == "nii.gz":
            image = nib.load(self.fixed_path)
            return list(image.header.get_data_shape())
        else:
            raise Exception("Unknown how to extract dimensions from filetype.")

    def set_paths(self):
        if not self["Collection"] or not self["Instance"]:
            logger.info("Collection and/or instance not set yet, can't determine paths.")
            return self
        
        collection = self["Collection"]
        instance = self["Instance"]
        extension = INSTANCES_CONFIG[collection]["extension"]
        folder = INSTANCES_CONFIG[collection]["folder"]
        fixed = f"{instance:02}_Fixed.{extension}"
        moving = f"{instance:02}_Moving.{extension}"

        self.fixed_path = INSTANCES_SRC / folder / "scans" / fixed
        self.moving_path = INSTANCES_SRC / folder / "scans" / moving

        self.fixedmask_path = None
        if INSTANCES_CONFIG[collection]["masks"]:
            self.fixedmask_path = INSTANCES_SRC / folder / "masks" / fixed
        if INSTANCES_CONFIG[collection]["landmarks"]:
            self.compute_tre = True
            self.lms_fixed_path = (
                INSTANCES_SRC / folder / "landmarks" / f"{fixed.split('.')[0]}.txt"
            )
            self.lms_moving_path = (
                INSTANCES_SRC / folder / "landmarks" / f"{moving.split('.')[0]}.txt"
            )
        else:
            self.compute_tre = False

        return self

    def __getitem__(self, key) -> Any:
        if key in self.params:
            return self.params[key]
        return False

    def __setitem__(self, key, value) -> None:
        self.params[key] = value

    def __str__(self) -> str:
        return f"{self.id[0]}_{self['Collection']}_{self['Instance']}_{self['Optimizer']}_{self.id[1]}".lower()

    def n_param(self, param: str, n: int = 2) -> List[str]:
        self[param] = [self[param] for i in range(n)]

    @staticmethod
    def parse_param(key, value):
        def esc(x):
            return f'"{x}"' if isinstance(x, str) else str(x)

        res = ""
        if isinstance(value, str):
            res += f"({key} {esc(value)})\n"
        elif isinstance(value, List):
            res += f"({key}"
            for elem in value:
                res += f" {esc(elem)}"
            res += f")\n"
        elif isinstance(value, bool):
            res += f"({key} {esc(str(value).lower())})\n"
        else:
            res += f"({key} {esc(value)})\n"
        return res

    @staticmethod
    def read_mhd(path: Path) -> Dict[str, Any]:
        mhd_properties = {}
        with path.open() as f:
            for line in f.readlines():
                line_split = line.split()
                key = line_split[0]
                values = line_split[2:]
                values = [Parameters.parse_mhd_value(x) for x in values]
                mhd_properties[key] = values
        return mhd_properties

    @staticmethod
    def parse_mhd_value(value):
        res = value
        try:
            res = int(value)
        except ValueError:
            try:
                res = float(value)
            except ValueError:
                return res
        return res


if __name__ == "__main__":
    params = (
        Parameters.from_base(mesh_size=5, seed=1)
        .asgd()
        .stopping_criteria(500)
        .instance(Collection.SYNTHETIC, 1)
    )
    params.write(Path())
