import sys
from pathlib import Path

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from thesispy.elastix_wrapper.parameters import Parameters, Collection
from thesispy.experiments.experiment import Experiment, run_experiment

params = (
    Parameters.from_base(mesh_size=5, seed=1, use_mask=False)
    .asgd()
    # .multi_resolution(1, p_sched=[3], downsampling=True)
    .stopping_criteria(100)
    .instance(Collection.SYNTHETIC, 1)
)

experiment = Experiment(params, "histotest")
run_experiment(experiment)
