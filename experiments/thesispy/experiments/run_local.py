import sys
from pathlib import Path

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from thesispy.elastix_wrapper.parameters import Parameters, Collection
from thesispy.experiments.experiment import Experiment, run_experiment

params = (
    Parameters.from_base(
        mesh_size=9,
        seed=1,
        metric="AdvancedNormalizedCorrelation",
        use_mask=True,
    )
    .asgd()
    .regularize(0.01)
    .multi_resolution(3, r_sched=[5, 4, 3], s_sched=[6, 2, 0], g_sched=[2, 2, 1])
    .stopping_criteria(iterations=[200, 400, 1000])
    .instance(Collection.LEARN, 1)
)

experiment = Experiment(params, "zandbak")
run_experiment(experiment)
