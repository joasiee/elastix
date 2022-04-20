import sys
from pathlib import Path

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from thesispy.elastix_wrapper.parameters import Parameters, Collection
from thesispy.experiments.experiment import Experiment, run_experiment

params = (
    Parameters.from_base(mesh_size=5, sampling_p=0.05)
    .multi_resolution(1, p_sched=[7, 7, 7])
    .multi_metric(weight0=1, weight1=100)
    .gomea()
    .instance(Collection.EMPIRE, 14)
    .stopping_criteria(iterations=[200])
)

experiment = Experiment(params, "zandbak")
run_experiment(experiment)
