import sys
from pathlib import Path

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from thesispy.elastix_wrapper.parameters import Parameters, Collection
from thesispy.experiments.experiment import Experiment, run_experiment

params = (
    Parameters.from_base(mesh_size=5, sampler="Full", seed=1)
    .multi_resolution(1, [4, 4, 4])
    .multi_metric(metric1="CorrespondingPointsEuclideanDistanceMetric", weight1=0.0)
    .asgd()
    .stopping_criteria(iterations=5000)
    .instance(Collection.LEARN, 1)
)

experiment = Experiment(params, "zandbak")
run_experiment(experiment)
