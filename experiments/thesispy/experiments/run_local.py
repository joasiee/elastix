import sys
from pathlib import Path

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from thesispy.elastix_wrapper.parameters import Parameters, Collection
from thesispy.experiments.experiment import Experiment, run_experiment

params = (
    Parameters.from_base(mesh_size=5, seed=1)
    .asgd()
    .stopping_criteria(0)
    .instance(Collection.LEARN, 1)
)

experiment = Experiment(params, "zandbak")
run_experiment(experiment)
