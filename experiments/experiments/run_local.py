import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from elastix_wrapper.parameters import Parameters, Collection
from experiments.experiment import Experiment, run_experiment

params = (
    Parameters.from_base(mesh_size=3, sampling_p=0.1, seed=5)
    .multi_resolution(1, [5, 5, 5])
    .gomea(fos=-6, partial_evals=True)
    .stopping_criteria(100)
    .instance(Collection.EMPIRE, 14)
)

# params.write(Path())
run_experiment(Experiment(params, "zandbak"))
