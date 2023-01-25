import sys
from pathlib import Path

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from thesispy.elastix_wrapper.parameters import Parameters, Collection
from thesispy.experiments.experiment import Experiment, run_experiment

params = (
    Parameters.from_base(mesh_size=5, seed=1)
    .asgd()
    .multi_resolution(1, p_sched=[3], downsampling=True)
    .stopping_criteria(3)
    .instance(Collection.LEARN, 1)
)

experiment = Experiment(params, "zandbak")
run_experiment(experiment)
