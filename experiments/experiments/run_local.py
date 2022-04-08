import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from elastix_wrapper.parameters import Parameters, Collection
from experiments.experiment import Experiment, run_experiment

params = (
    Parameters.from_base(mesh_size=7, sampler="Full")
    .multi_resolution(3)
    .asgd()
    .stopping_criteria(100)
    .instance(Collection.EXAMPLES, 1)
)

# params.write(Path())
run_experiment(Experiment(params, "zandbak"))
