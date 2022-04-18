import sys
from pathlib import Path

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from thesispy.elastix_wrapper.parameters import Parameters, Collection
from thesispy.experiments.experiment import Experiment, run_experiment

params = (
    Parameters.from_base(mesh_size=5, sampler="RandomCoordinate")
    .gomea()
    .stopping_criteria(10)
    .instance(Collection.EXAMPLES, 1)
)

experiment = Experiment(params, "zandbak")
run_experiment(experiment)