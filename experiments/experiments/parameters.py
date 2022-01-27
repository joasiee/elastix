from __future__ import annotations

import json
from math import ceil
import os
from PIL import Image
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

BASE_PARAMS_PATH = Path("resources", "base_params.json")
INSTANCE_CONFIG_PATH = Path("resources", "instances.json")
INSTANCES_CONFIG: Dict[str, str] = {}
INSTANCES_SRC = Path(os.environ.get("INSTANCES_SRC"))

with INSTANCE_CONFIG_PATH.open() as f:
    INSTANCES_CONFIG = json.loads(f.read())


class Collection(str, Enum):
    EMPIRE = "EMPIRE"
    EXAMPLES = "EXAMPLES"


class Parameters:
    def __init__(
        self,
        metric: str = "AdvancedMeanSquares",
        sampler: str = "Random",
        sampling_p: float = 0.05,
        mesh_size: List[int] | int = 20,
    ) -> None:
        with BASE_PARAMS_PATH.open() as f:
            self.params: Dict[str, Any] = json.loads(f.read())
        self["Metric"] = metric
        self["ImageSampler"] = sampler
        self["SamplingPercentage"] = sampling_p
        self["MeshSize"] = mesh_size

    def instance(self, collection: Collection, instance: int) -> Parameters:
        folder = INSTANCES_CONFIG[collection.value]["folder"]
        extension = INSTANCES_CONFIG[collection.value]["extension"]
        fixed = Path(folder, f"{instance:02}_Fixed.{extension}")
        moving = Path(folder, f"{instance:02}_Moving.{extension}")
        self["Collection"] = collection
        self["Instance"] = instance
        self.fixed_path = INSTANCES_SRC.joinpath(fixed)
        self.moving_path = INSTANCES_SRC.joinpath(moving)
        return self.calc_voxel_params()

    def multi_metric(
        self,
        metric1: str = "AdvancedMeanSquares",
        metric2: str = "TransformBendingEnergyPenalty",
        weight1: float = 0.8,
        weight2: float = 0.2,
    ) -> Parameters:
        self["Registration"] = "MultiMetricMultiResolutionRegistration"
        self["FixedImagePyramid"] = [
            "FixedSmoothingImagePyramid",
            "FixedSmoothingImagePyramid",
        ]
        self["MovingImagePyramid"] = [
            "MovingSmoothingImagePyramid",
            "MovingSmoothingImagePyramid",
        ]
        self["Interpolator"] = ["LinearInterpolator", "LinearInterpolator"]
        self["Metric"] = [metric1, metric2]
        self["Metric0Weight"] = weight1
        self["Metric1Weight"] = weight2
        return self

    def multi_resolution(
        self, n: int = 3, p_sched: List[int] = None, g_sched: List[float] = None
    ) -> Parameters:
        self["NumberOfResolutions"] = n
        self["ImagePyramidSchedule"] = p_sched
        self["GridSpacingSchedule"] = g_sched
        return self

    def optimizer(self, optim: str, params: Dict[str, Any] = None) -> Parameters:
        self["Optimizer"] = optim
        if params is not None:
            for key, value in params.items():
                self[key] = value
        return self

    def stopping_criteria(
            self,
            iterations: List[int] | int = None,
            evals: List[int] | int = None,
            max_time_s: int = 0):
        self["MaximumNumberOfIterations"] = iterations
        self["MaxNumberOfEvaluations"] = evals
        self["MaxTimeSeconds"] = max_time_s
        return self

    def gomea(
        self,
        fos: int = None,
        pop_size: List[int] | int = None,
        partial_evals: bool = None,
    ) -> Parameters:
        return self.optimizer(
            "GOMEA",
            {
                "FosElementSize": fos,
                "BasePopulationSize": pop_size,
                "PartialEvaluations": partial_evals,
            },
        )

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
        if not isinstance(self["MeshSize"], List):
            self["MeshSize"] = [self["MeshSize"]]
        voxel_dims = self.get_voxel_dimensions()
        voxel_spacings = []
        total_samples = 1
        for i, voxel_dim in enumerate(voxel_dims):
            voxel_spacings.append(
                ceil(voxel_dim / self["MeshSize"]
                     [min(i, len(self["MeshSize"]) - 1)])
            )
            total_samples *= voxel_dim

        self["FinalGridSpacingInVoxels"] = voxel_spacings
        # self["NumberOfSpatialSamples"] = int(total_samples * self["SamplingPercentage"])
        return self

    def get_voxel_dimensions(self) -> List[int]:
        match INSTANCES_CONFIG[self["Collection"].value]["extension"]:
            case 'mhd':
                return Parameters.read_mhd(self.fixed_path)["DimSize"]
            case 'png':
                image = Image.open(str(self.fixed_path))
                return list(image.size)
            case _:
                raise Exception(
                    "Unknown how to extract dimensions from filetype.")

    def __getitem__(self, key) -> Any:
        return self.params[key]

    def __setitem__(self, key, value) -> None:
        self.params[key] = value

    def __str__(self) -> str:
        return f"{datetime.now()} - {self['Collection']}: {self['Instance']} - {self['Optimizer']}"

    @staticmethod
    def parse_param(key, value):
        def esc(x): return f'"{x}"' if isinstance(x, str) else str(x)
        res = ""
        if isinstance(value, str):
            res += f"({key} {esc(value)})\n"
        elif isinstance(value, List):
            res += f"({key}"
            for elem in value:
                res += f" {esc(elem)}"
            res += f")\n"
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
