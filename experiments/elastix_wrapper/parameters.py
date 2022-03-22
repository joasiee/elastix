from __future__ import annotations

import json
from math import ceil
import os
import uuid
from PIL import Image
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

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
        params
    ) -> None:
        self.params = params
        self.id = uuid.uuid1()

    @classmethod
    def from_base(cls,
                  metric: str = "AdvancedMeanSquares",
                  sampler: str = "RandomCoordinate",
                  sampling_p: float = 0.02,
                  mesh_size: List[int] | int = 12,
                  seed: int = None,
                  write_img=False):
        with BASE_PARAMS_PATH.open() as f:
            params: Dict[str, Any] = json.loads(f.read())
        params["Metric"] = metric
        params["ImageSampler"] = sampler
        params["SamplingPercentage"] = sampling_p
        params["MeshSize"] = mesh_size
        params["RandomSeed"] = seed
        params["WriteResultImage"] = write_img
        return cls(params)

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
        metric1: str = "TransformBendingEnergyPenalty",
        weight0: float = 1.0,
        weight1: float = 1.0,
    ) -> Parameters:
        self["Registration"] = "MultiMetricMultiResolutionRegistration"
        self.n_param("FixedImagePyramid", 2)
        self.n_param("MovingImagePyramid", 2)
        self.n_param("Interpolator", 2)
        self.n_param("ImageSampler", 2)
        self["Metric"] = [metric0, metric1]
        self["Metric0Weight"] = weight0
        self["Metric1Weight"] = weight1
        return self

    def multi_resolution(
            self, n: int = 3, p_sched: List[int] = None, g_sched: List[int] = None) -> Parameters:
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
            max_time_s: int = 0,
            fitness_var: float = 1e-9):
        self["MaximumNumberOfIterations"] = iterations
        self["MaxNumberOfEvaluations"] = evals
        self["MaxTimeSeconds"] = max_time_s
        self["FitnessVarianceTolerance"] = fitness_var
        return self

    def gomea(
        self,
        fos: int = None,
        pop_size: List[int] | int = None,
        partial_evals: bool = None,
    ) -> Parameters:
        if partial_evals:
            self["NewSamplesEveryIteration"] = "false"
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
        voxel_dims = self.get_voxel_dimensions()
        if not isinstance(self["MeshSize"], List):
            self["MeshSize"] = [self["MeshSize"]
                                for _ in range(len(voxel_dims))]
        if "ImagePyramidSchedule" not in self.params or not self["ImagePyramidSchedule"]:
            self["ImagePyramidSchedule"] = [
                2**n for n in range(self["NumberOfResolutions"] - 1, -1, -1) for _ in range(len(voxel_dims))]
        voxel_spacings = []
        total_samples = [1] * self["NumberOfResolutions"]
        for i, voxel_dim in enumerate(voxel_dims):
            voxel_spacings.append(ceil(voxel_dim / self["MeshSize"][i]))
            for n in range(len(total_samples)):
                total_samples[n] *= int(voxel_dim /
                                        self["ImagePyramidSchedule"][n*len(voxel_dims)+i])

        self["FinalGridSpacingInVoxels"] = voxel_spacings
        self["NumberOfSpatialSamples"] = [
            int(x * self["SamplingPercentage"]) for x in total_samples]
        return self

    def get_voxel_dimensions(self) -> List[int]:
        extension = INSTANCES_CONFIG[self["Collection"]]["extension"]
        if extension == 'mhd':
            return Parameters.read_mhd(self.fixed_path)["DimSize"]
        elif extension == 'png':
            image = Image.open(str(self.fixed_path))
            return list(image.size)
        else:
            raise Exception(
                "Unknown how to extract dimensions from filetype.")

    def set_paths(self):
        if "Collection" in self.params and "Instance" in self.params:
            collection = self["Collection"]
            instance = self["Instance"]
            extension = INSTANCES_CONFIG[collection]["extension"]
            folder = INSTANCES_CONFIG[collection]["folder"]
            fixed = f"{instance:02}_Fixed.{extension}"
            moving = f"{instance:02}_Moving.{extension}"
            self.fixed_path = INSTANCES_SRC / folder / "scans" / fixed
            self.fixedmask_path = INSTANCES_SRC / folder / "masks" / \
                fixed if INSTANCES_CONFIG[collection]["masks"] else None
            self.moving_path = INSTANCES_SRC / folder / "scans" / moving
        return self

    def __getitem__(self, key) -> Any:
        return self.params[key]

    def __setitem__(self, key, value) -> None:
        self.params[key] = value

    def __str__(self) -> str:
        return f"{self['Collection']}_{self['Instance']}_{self['Optimizer']}_{self.id}".lower()

    def n_param(self, param: str, n: int = 2) -> List[str]:
        self[param] = [self[param] for i in range(n)]

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
    sched = [9, 9, 9, 6, 6, 6, 5, 5, 5, 4, 4, 4]
    params = Parameters.from_base(mesh_size=10).gomea(
    ).multi_resolution(4, p_sched=sched).multi_metric().instance(Collection.EMPIRE, 7)
    params.write(Path())
