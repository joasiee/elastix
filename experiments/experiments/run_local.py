import sys
from pathlib import Path

import numpy as np

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from elastix_wrapper.parameters import Parameters, Collection
from experiments.experiment import Experiment, run_experiment


params = (
    Parameters.from_base(mesh_size=2, seed=1523, sampling_p=0.2)
    .multi_resolution(1, p_sched=[7, 7, 7])
    .multi_metric()
    .asgd()
    .instance(Collection.EMPIRE, 26)
    .stopping_criteria(iterations=[20000])
)

run_experiment(Experiment(params, "asgd_26_sampling"))