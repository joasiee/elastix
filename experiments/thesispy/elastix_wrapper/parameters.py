from __future__ import annotations
from datetime import datetime

import json
import logging
from math import ceil
import time
from PIL import Image
import nibabel as nib
from pathlib import Path
from typing import Any, Dict, List
from thesispy.definitions import *

logger = logging.getLogger("Parameters")


class Parameters:
    def __init__(self, params) -> None:
        self.params = params
        self.id = [str(int(time.time())), str(datetime.now().microsecond)]

    @classmethod
    def from_base(
        cls,
        metric: str = "AdvancedMeanSquares",
        mesh_size: List[int] | int = 4,
        use_mask: bool = None,
        use_missedpixel_penalty: bool = None,
        seed: int = None,
    ):
        with BASE_PARAMS_PATH.open() as f:
            params: Dict[str, Any] = json.loads(f.read())
        return cls(params).args({"Metric": metric, "MeshSize": mesh_size, "UseMask": use_mask, "UseMissedPixelPenalty": use_missedpixel_penalty, "RandomSeed": seed})

    @classmethod
    def from_json(cls, jsondump):
        params = json.loads(jsondump)
        return cls(params).set_paths()

    def instance(self, collection: Collection, instance: int) -> Parameters:
        self["Collection"] = collection.value
        self["Instance"] = instance
        self.set_paths()
        return self.calc_voxel_params()

    def regularize(
        self,
        weight: float,
        analytic: bool = True,
    ) -> Parameters:
        self.n_param("FixedImagePyramid", 2)
        self.n_param("MovingImagePyramid", 2)
        self.n_param("Interpolator", 2)
        self.n_param("ImageSampler", 2)
        regularize_metric = "TransformBendingEnergyPenaltyAnalytic" if analytic else "TransformBendingEnergyPenalty"

        return self.args(
            {
                "Registration": "MultiMetricMultiResolutionRegistration",
                "Metric": [self["Metric"], regularize_metric],
                "Metric0Weight": 1.0,
                "Metric1Weight": weight,
            }
        )

    def multi_resolution(
        self, n: int = 3, p_sched: List[int] = None, g_sched: List[int] = None, downsampling: bool = False
    ) -> Parameters:
        args = {
            "NumberOfResolutions": n,
            "ImagePyramidSchedule": p_sched,
            "GridSpacingSchedule": g_sched,
        }
        if not downsampling:
            args["FixedImagePyramid"] = "FixedSmoothingImagePyramid"
            args["MovingImagePyramid"] = "MovingSmoothingImagePyramid"

        return self.args(args)

    def stopping_criteria(
        self,
        iterations: List[int] | int = None,
        evals: List[int] | int = None,
        pixel_evals: List[int] | int = None,
        max_time_s: int = 0,
        fitness_var: float = None,
    ):
        return self.args(
            {
                "MaximumNumberOfIterations": iterations,
                "MaxNumberOfEvaluations": evals,
                "MaxNumberOfPixelEvaluations": pixel_evals,
                "MaxTimeSeconds": max_time_s,
                "FitnessVarianceTolerance": fitness_var,
            }
        )

    def gomea(
        self,
        fos: LinkageType = LinkageType.CP_MARGINAL,
        pop_size: List[int] | int = None,
        shrinkage: bool = None,
        use_constraints: bool = None,
        contraints_threshold: float = 0.0,
        min_set_size: int = None,
        max_set_size: int = None,
        hybrid: bool = None,
        tau_asgd: float = None,
        asgd_iterations: int = None,
    ) -> Parameters:
        pevals = False if fos == LinkageType.FULL else True
        static_linkage_type = 0
        if fos in STATIC_LINKAGE_MAPPING:
            static_linkage_type = STATIC_LINKAGE_MAPPING[fos]
        return self.args(
            {
                "Optimizer": "GOMEA",
                "OptimizerName": "RV-GOMEA",
                "FosElementSize": fos.value,
                "BasePopulationSize": pop_size,
                "PartialEvaluations": pevals,
                "UseShrinkage": shrinkage,
                "UseConstraints": use_constraints,
                "MissedPixelConstraintThreshold": contraints_threshold * 100.0,
                "StaticLinkageType": static_linkage_type,
                "StaticLinkageMinSetSize": min_set_size,
                "StaticLinkageMaxSetSize": max_set_size,
                "UseASGD": hybrid,
                "TauASGD": tau_asgd,
                "NumberOfASGDIterations": asgd_iterations,
            }
        )

    def asgd(self, params: Dict[str, Any] = None):
        return self.args(
            {
                "Optimizer": "AdaptiveStochasticGradientDescent",
                "OptimizerName": "ASGD",
            },
            params,
        )

    def debug(self):
        return self.args(
            {
                "WritePyramidImagesAfterEachResolution": True,
                "WriteSamplesEveryIteration": True,
                "WriteControlPointsEveryIteration": True,
                "WriteExtraOutput": True,
            }
        )

    def sampler(self, sampler, pct: float = 0.05):
        return self.args({"ImageSampler": sampler, "SamplingPercentage": pct})

    def args(self, params: Dict[str, Any], extra_params: Dict[str, Any] = None) -> Parameters:
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
            self["GridSpacingSchedule"] = [i for i in range(self["NumberOfResolutions"], 0, -1)]

        if not self["ImagePyramidSchedule"]:
            self["ImagePyramidSchedule"] = [
                n for n in range(self["NumberOfResolutions"], 0, -1) for _ in range(len(voxel_dims))
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
                total_samples[n] *= int(voxel_dim / self["ImagePyramidSchedule"][n * len(voxel_dims) + i])

        if "FinalGridSpacingInVoxels" not in self.params:
            self["FinalGridSpacingInVoxels"] = voxel_spacings

        self["NumberOfSpatialSamples"] = total_samples
        if self["ImageSampler"] != "Full" and "Full" not in self["ImageSampler"]:
            self["NumberOfSpatialSamples"] = [
                int(x * self["SamplingPercentage"]) for x in self["NumberOfSpatialSamples"]
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
            self.lms_fixed_path = INSTANCES_SRC / folder / "landmarks" / f"{fixed.split('.')[0]}.txt"

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
        Parameters.from_base(mesh_size=2, seed=1, metric="AdvancedMeanSquares", use_missedpixel_penalty=True)
        .gomea(LinkageType.CP_MARGINAL, contraints_threshold=0.1, pop_size=10)
        .regularize(0.01)
        .multi_resolution(2, g_sched=[1, 1])
        .stopping_criteria(5)
        .instance(Collection.SYNTHETIC, 1)
    )
    params.write(Path())
