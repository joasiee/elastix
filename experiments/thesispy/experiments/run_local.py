import sys
from pathlib import Path

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from thesispy.elastix_wrapper.parameters import Parameters, Collection
from thesispy.experiments.experiment import Experiment, run_experiment

params = (
    Parameters.from_base(mesh_size=6, metric="AdvancedNormalizedCorrelation", seed=1, use_mask=True)
    .asgd()
    .regularize(0.01)
    .multi_resolution(3, p_sched=[6, 4, 2])
    .stopping_criteria(200)
    .instance(Collection.LEARN, 1)
)

experiment = Experiment(params, "zandbak")
run_experiment(experiment)
