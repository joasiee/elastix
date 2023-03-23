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
    """Class for building parameter files to be used by elastix."""

    def __init__(self, params) -> None:
        """Initialize using predefined dictionary of parameters."""
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
        """Initialize from base parameters file.

        Args:
            metric (str, optional): Metric to use. Defaults to "AdvancedMeanSquares".
            mesh_size (List[int] | int, optional): Mesh size. Defaults to 4.
            use_mask (bool, optional): Use mask or not. Defaults to None.
            use_missedpixel_penalty (bool, optional): Use missed pixel penalty or not. Defaults to None.
            seed (int, optional): Random seed. Defaults to None.
        """
        with BASE_PARAMS_PATH.open() as f:
            params: Dict[str, Any] = json.loads(f.read())
        return cls(params).args(
            {
                "Metric": metric,
                "MeshSize": mesh_size,
                "UseMask": use_mask,
                "UseMissedPixelPenalty": use_missedpixel_penalty,
                "RandomSeed": seed,
            }
        )

    @classmethod
    def from_json(cls, jsondump):
        """Initialize from json."""
        params = json.loads(jsondump)
        return cls(params).set_paths()

    def instance(self, collection: Collection, instance: int) -> Parameters:
        """Set collection and instance.

        Args:
            collection (Collection): Collection to use.
            instance (int): Instance of collection to use.
        """
        self["Collection"] = collection.value
        self["Instance"] = instance
        self.set_paths()
        return self.calc_voxel_params()

    def regularize(
        self,
        weight: float,
        analytic: bool = True,
    ) -> Parameters:
        """Use regularization by adding a penalty term to the metric.

        Args:
            weight (float): Weight of penalty term.
            analytic (bool, optional): Use analytic or numerical penalty implementation. Defaults to analytical.
        """
        if weight == 0:
            return self.args({"Metric0Weight": 1.0, "Metric1Weight": weight})

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
        self,
        n: int = 3,
        r_sched: List[int] = None,
        s_sched: List[int] = None,
        g_sched: List[int] = None,
        downsampling: bool = True,
        smoothing: bool = True,
    ) -> Parameters:
        """Use multi-resolution scheme.

        Args:
            n (int, optional): Number of resolutions. Defaults to 3.
            r_sched (List[int], optional): Rescale schedule. Defaults to None.
            s_sched (List[int], optional): Smoothing schedule. Defaults to None.
            g_sched (List[int], optional): Grid spacing schedule. Defaults to None.
            downsampling (bool, optional): Use downsampling, if disabled an identity rescale schedule is used. Defaults to True.
            smoothing (bool, optional): Use smoothing, if disabled an identity smoothing schedule is used. Defaults to True.
        """
        args = {
            "NumberOfResolutions": n,
            "ImagePyramidRescaleSchedule": r_sched,
            "ImagePyramidSmoothingSchedule": s_sched,
            "GridSpacingSchedule": g_sched,
            "Downsampling": downsampling,
            "Smoothing": smoothing,
        }

        return self.args(args)

    def stopping_criteria(
        self,
        iterations: List[int] | int = None,
        evals: List[int] | int = None,
        pixel_evals: List[int] | int = None,
        max_time_s: int = 0,
        fitness_var: float = None,
    ):
        """Set stopping criteria for registration.

        Args:
            iterations (List[int] | int, optional): Maximum number of iterations. Defaults to None.
            evals (List[int] | int, optional): Maximum number of objective function evaluations. Defaults to None.
            pixel_evals (List[int] | int, optional): Maximum number of pixel evaluations. Defaults to None.
            max_time_s (int, optional): Maximum time in seconds. Defaults to 0 (i.e. no time constraint).
            fitness_var (float, optional): Fitness variance tolerance. Defaults to None.
        """
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
        missed_pixel_threshold: float = None,
        compute_folds_constraints: bool = None,
        min_set_size: int = None,
        max_set_size: int = None,
        hybrid: bool = None,
        tau_asgd: float = None,
        alpha_asgd: float = None,
        beta_asgd: float = None,
        asgd_iterations: int = None,
        asgd_iterations_max: int = None,
        asgd_iterations_min: int = None,
        asgd_iterations_offset: int = None,
        redis_method: RedistributionMethod = None,
        it_schedule: IterationSchedule = None,
    ) -> Parameters:
        """Use the GOMEA optimizer for the registration.

        Args:
            fos (LinkageType, optional): Linkage type. Defaults to LinkageType.CP_MARGINAL.
            pop_size (List[int] | int, optional): Population size. Defaults to None, which means that the optimizer calculates a suitable population size given the registration parameters.
            shrinkage (bool, optional): Use shrinkage. Defaults to None.
            use_constraints (bool, optional): Use constraints. Defaults to None.
            missed_pixel_threshold (float, optional): Missed pixel constraint threshold [0, 1]. Defaults to None.
            compute_folds_constraints (bool, optional): Compute folds constraints (GOMEA-FC). Defaults to None.
            min_set_size (int, optional): Minimum set size for static linkage. Defaults to None.
            max_set_size (int, optional): Maximum set size for static linkage. Defaults to None.
            hybrid (bool, optional): Use hybrid local search with ASGD (GOMEA-LS). Defaults to None.
            tau_asgd (float, optional): ASGD tau parameter (GOMEA-LS). Defaults to None.
            alpha_asgd (float, optional): ASGD alpha parameter (GOMEA-LS). Defaults to None.
            beta_asgd (float, optional): ASGD beta parameter (GOMEA-LS). Defaults to None.
            asgd_iterations (int, optional): ASGD iterations (GOMEA-LS). Defaults to None.
            asgd_iterations_max (int, optional): ASGD maximum iterations (GOMEA-LS). Defaults to None.
            asgd_iterations_min (int, optional): ASGD minimum iterations (GOMEA-LS). Defaults to None.
            asgd_iterations_offset (int, optional): ASGD iterations offset (GOMEA-LS). Defaults to None.
            redis_method (RedistributionMethod, optional): Redistribution method (GOMEA-LS). Defaults to None.
            it_schedule (IterationSchedule, optional): Iteration schedule (GOMEA-LS). Defaults to None.
        """
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
                "MissedPixelConstraintThreshold": missed_pixel_threshold,
                "ComputeControlPointFolds": compute_folds_constraints,
                "StaticLinkageType": static_linkage_type,
                "StaticLinkageMinSetSize": min_set_size,
                "StaticLinkageMaxSetSize": max_set_size,
                "UseASGD": hybrid,
                "TauASGD": tau_asgd,
                "AlphaASGD": alpha_asgd,
                "BetaASGD": beta_asgd,
                "NumberOfASGDIterations": asgd_iterations,
                "MaxNumberOfASGDIterations": asgd_iterations_max,
                "MinNumberOfASGDIterations": asgd_iterations_min,
                "NumberOfASGDIterationsOffset": asgd_iterations_offset,
                "RedistributionMethod": redis_method.value if redis_method is not None else None,
                "ASGDIterationSchedule": it_schedule.value if it_schedule is not None else None,
            }
        )

    def asgd(self, params: Dict[str, Any] = None):
        """Use the ASGD optimizer for the registration."""
        return self.args(
            {
                "Optimizer": "AdaptiveStochasticGradientDescent",
                "OptimizerName": "ASGD",
            },
            params,
        )

    def rigid(self):
        """Use the rigid transform for the registration."""
        return self.args(
            {
                "Transform": "EulerTransform",
                "AutomaticScalesEstimation": True,
                "AutomaticTransformInitialization": True,
                "AutomaticTransformInitializationMethod": "Origins",
            }
        )

    def affine(self):
        """Use the affine transform for the registration."""
        self.rigid()
        return self.args({"Transform": "SimilarityTransform"})

    def debug(self):
        """Write additional debug output to the output folder of the registration"""
        return self.args(
            {
                "WritePyramidImagesAfterEachResolution": True,
                "WriteSamplesEveryIteration": True,
                "WriteControlPointsEveryIteration": True,
                "WriteExtraOutput": True,
            }
        )

    def sampler(self, sampler, pct: float = 0.05):
        """Voxel sampler to use for the registration.

        Args:
            sampler (VoxelSampler): Voxel sampler to use.
            pct (float, optional): Percentage of voxels to sample. Defaults to 0.05.
        """
        return self.args({"ImageSampler": sampler, "SamplingPercentage": pct})

    def args(self, params: Dict[str, Any], extra_params: Dict[str, Any] = None) -> Parameters:
        """Add parameters to the registration parameters.

        Args:
            params (Dict[str, Any]): Parameters to add.
            extra_params (Dict[str, Any], optional): Extra parameters to add (lazy code). Defaults to None.
        """
        for key, value in params.items():
            self[key] = value
        if extra_params is not None:
            for key, value in extra_params.items():
                self[key] = value
        return self

    def prune(self):
        """Remove all parameters that are None."""
        self.params = {k: v for k, v in self.params.items() if v is not None}

    def write(self, dir: Path, suffix=1) -> None:
        """Write the parameters to a parameter file in the format for elastix."""
        self.prune()
        out_file = dir.joinpath(Path(f"params_{suffix}.txt"))
        with open(str(out_file), "w+") as f:
            for key, value in self.params.items():
                f.write(Parameters.parse_param(key, value))
        return out_file

    def calc_voxel_params(self) -> Parameters:
        """Calculate the voxel spacings for the B-spline transform and the number of spatial samples for the image sampler."""
        voxel_dims = self.get_voxel_dimensions()

        if not isinstance(self["MeshSize"], List):
            self["MeshSize"] = [self["MeshSize"] for _ in range(len(voxel_dims))]

        self.calc_multiresolution_schedules(voxel_dims)

        voxel_spacings = []
        total_samples = [1] * self["NumberOfResolutions"]
        for i, voxel_dim in enumerate(voxel_dims):
            voxel_spacings.append(ceil(voxel_dim / self["MeshSize"][i]))
            for n in range(len(total_samples)):
                total_samples[n] *= int(voxel_dim / self["ImagePyramidRescaleSchedule"][n * len(voxel_dims) + i])

        if "FinalGridSpacingInVoxels" not in self.params:
            self["FinalGridSpacingInVoxels"] = voxel_spacings

        self["NumberOfSpatialSamples"] = total_samples
        if self["ImageSampler"] != "Full" and "Full" not in self["ImageSampler"]:
            self["NumberOfSpatialSamples"] = [
                int(x * self["SamplingPercentage"]) for x in self["NumberOfSpatialSamples"]
            ]

        return self

    def calc_multiresolution_schedules(self, voxel_dims):
        """Helper function which reshapes the ImagePyramidRescaleSchedule and ImagePyramidSmoothingSchedule parameters to the correct format for elastix."""
        if not self["Downsampling"]:
            self["ImagePyramidRescaleSchedule"] = [1 for _ in range(self["NumberOfResolutions"] * len(voxel_dims))]
        elif self["ImagePyramidRescaleSchedule"] is None:
            self["ImagePyramidRescaleSchedule"] = [
                2 ** (self["NumberOfResolutions"] - i - 1)
                for i in range(self["NumberOfResolutions"])
                for _ in range(len(voxel_dims))
            ]
        elif len(self["ImagePyramidRescaleSchedule"]) == self["NumberOfResolutions"]:
            self["ImagePyramidRescaleSchedule"] = [
                self["ImagePyramidRescaleSchedule"][i]
                for i in range(self["NumberOfResolutions"])
                for _ in range(len(voxel_dims))
            ]

        if not self["Smoothing"]:
            self["ImagePyramidSmoothingSchedule"] = [0 for _ in range(self["NumberOfResolutions"] * len(voxel_dims))]
        elif (
            self["ImagePyramidSmoothingSchedule"] is not None
            and len(self["ImagePyramidSmoothingSchedule"]) == self["NumberOfResolutions"]
        ):
            self["ImagePyramidSmoothingSchedule"] = [
                self["ImagePyramidSmoothingSchedule"][i]
                for i in range(self["NumberOfResolutions"])
                for _ in range(len(voxel_dims))
            ]

    def get_voxel_dimensions(self) -> List[int]:
        """Get the voxel dimensions of the fixed image."""
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
        """Set the paths to the images, masks, and landmarks."""
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

    def n_param(self, param: str, n: int = 2) -> List[str]:
        self[param] = [self[param] for i in range(n)]

    def __getitem__(self, key) -> Any:
        if key in self.params:
            return self.params[key]
        return False

    def __setitem__(self, key, value) -> None:
        self.params[key] = value

    def __str__(self) -> str:
        return f"{self.id[0]}_{self['Collection']}_{self['Instance']}_{self['Optimizer']}_{self.id[1]}".lower()

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
        Parameters.from_base(mesh_size=4, seed=1, use_mask=True)
        .gomea()
        .multi_resolution(1, r_sched=[5])
        .stopping_criteria(iterations=3)
        .instance(Collection.LEARN, 1)
    )
    params.write(Path())
