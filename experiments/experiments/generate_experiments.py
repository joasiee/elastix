import logging

import numpy as np
from elastix_wrapper.parameters import Parameters, Collection
from experiments.experiment import Experiment, ExperimentQueue

logger = logging.getLogger("ParetoFront")

def sampling_p_range(instance: int, project: str):
    for sampling_p in list(np.arange(0.01, 0.11, 0.01)):
        for seed in [1, 2, 3, 4, 5]:
            params = (
                Parameters.from_base(mesh_size=3, seed=seed, sampling_p=sampling_p)
                .multi_resolution(1, p_sched=[5, 5, 5])
                .gomea(fos=-6, partial_evals=True)
                .instance(Collection.EMPIRE, instance)
                .stopping_criteria(iterations=[50])
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
                .gomea(partial_evals=True, fos=-6)
                .stopping_criteria(50)
                .instance(Collection.EMPIRE, instance)
            )
            yield Experiment(params, project)

if __name__ == "__main__":
    queue = ExperimentQueue()
    for experiment in sampling_p_range(16, "sampling_experiment2"):
        queue.push(experiment)
