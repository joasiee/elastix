from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

BASE_PARAMS_PATH = Path('experiments', 'resources', 'base_params.json')

class Parameters:
    def __init__(
        self,
        metric: str = "AdvancedMeanSquares",
        sampler: str = "RandomSampler",
        sampling_p: float = 0.05,
        g_spacing: List[float] = [10.0],
    ) -> None:
        with BASE_PARAMS_PATH.open() as f:
            self.params: Dict[str, Any] = json.loads(f.read())
        self["Metric"] = metric
        self["ImageSampler"] = sampler
        self["SamplingPercentage"] = sampling_p
        self["FinalGridSpacingInPhysicalUnits"] = g_spacing

    def __getitem__(self, key) -> Any:
        return self.params[key]

    def __setitem__(self, key, value) -> None:
        self.params[key] = value

    def prune(self):
        self.params = {k: v for k, v in self.params.items() if v is not None}

    def write(self, dir: Path) -> None:
        self.prune()
        out_file = dir.joinpath(Path("params.txt"))
        with open(str(out_file), "w+") as f:
            for key, value in self.params.items():
                f.write(Parameters.parse_param(key, value))

    @staticmethod
    def parse_param(key, value):
        esc = lambda x: f'"{x}"' if isinstance(x, str) else str(x)
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

    def optimizer(self, optim: str, params: Dict[str, Any]) -> Parameters:
        self["Optimizer"] = optim
        for key, value in params.items():
            self[key] = value
        return self

    def gomea(
        self,
        iterations: List[int] = None,
        evals: List[int] = None,
        fos: int = None,
        pop_size: List[int] = None,
        partial_evals: bool = None,
    ) -> Parameters:
        return self.optimizer(
            "GOMEA",
            {
                "MaximumNumberOfIterations": iterations,
                "MaxNumberOfEvaluations": evals,
                "FosElementSize": fos,
                "BasePopulationSize": pop_size,
                "PartialEvaluations": partial_evals,
            },
        )