import logging

import numpy as np
from elastix_wrapper.parameters import Parameters, Collection
from experiments.experiment import Experiment, ExperimentQueue

logger = logging.getLogger("ParetoFront")

def sampling_p_range(instance: int, project: str):
    for sampling_p in [0.01, 0.02, 0.03] + list(np.arange(0.05, 0.55, 0.05)):
        for seed in [2356, 3487, 4942, 1432]:
            params = (
                Parameters.from_base(mesh_size=2, seed=seed, sampling_p=sampling_p)
                .multi_resolution(1, p_sched=[7, 7, 7])
                .multi_metric()
                .gomea()
                .instance(Collection.EMPIRE, 26)
                .stopping_criteria(iterations=[100])
            )
            yield Experiment(params, project)

def pd_pop_experiment(project):
    for pop_size in [2**x for x in range(5, 10)]:
        for seed in [25324, 65984, 24137, 96897, 86458]:
            params = (
                Parameters.from_base(mesh_size=2, sampler="Full", seed=seed)
                .gomea(pop_size=pop_size)
                .instance(Collection.EXAMPLES, 1)
                .stopping_criteria(iterations=100)
            )
            yield Experiment(params, project)

def convergence_experiment(project):
    for instance in [16, 17, 14]:
        for seed in [1, 2, 3, 4, 5]:
            sched = [7, 7 ,7] if instance == 14 else [5, 5, 5]
            params = (
                Parameters.from_base(mesh_size=3, sampling_p=0.1, seed=seed)
                .multi_resolution(1, sched)
                .gomea()
                .stopping_criteria(100)
                .instance(Collection.EMPIRE, instance)
            )
            yield Experiment(params, project)

if __name__ == "__main__":
    queue = ExperimentQueue()
    for experiment in convergence_experiment("convergence_experiment"):
        queue.push(experiment)
