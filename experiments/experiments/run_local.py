import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from elastix_wrapper.watchdog import SaveStrategyWandb
from elastix_wrapper import wrapper
from elastix_wrapper.parameters import Parameters, Collection
from experiments.experiment import Experiment

params = (
    Parameters.from_base(mesh_size=5, sampler="Full")
    .asgd()
    .stopping_criteria(5000)
    .instance(Collection.EXAMPLES, 1)
)

experiment = Experiment(params, "zandbak")
run_dir = Path("output") / experiment.project / str(experiment.params)
sv_strat = SaveStrategyWandb(experiment, run_dir, 100)
wrapper.run(experiment.params, run_dir, sv_strat)