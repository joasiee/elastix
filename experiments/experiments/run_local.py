import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from elastix_wrapper.parameters import Parameters, Collection
from experiments.experiment import Experiment, run_experiment

params = (
    Parameters.from_base(mesh_size=2, seed=1523, sampling_p=0.05)
    .multi_resolution(2)
    .asgd()
    .instance(Collection.EMPIRE, 26)
    .stopping_criteria(iterations=[5, 1])
)

run_experiment(Experiment(params, "zandbak"))