import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from elastix_wrapper.parameters import Parameters, Collection
from experiments.experiment import Experiment, run_experiment

params = (
    Parameters.from_base(mesh_size=5, sampler="Full")
    .asgd()
    .stopping_criteria(2000)
    .instance(Collection.EXAMPLES, 1)
)

# params.write(Path())
run_experiment(Experiment(params, "zandbak"))
