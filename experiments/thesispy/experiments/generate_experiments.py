import logging
import random

import numpy as np
from thesispy.elastix_wrapper.parameters import Parameters, Collection
from thesispy.experiments.experiment import Experiment, ExperimentQueue

logger = logging.getLogger("ParetoFront")


def sampling_p_range(instance: int, project: str):
    for sampling_p in list(np.arange(0.01, 0.11, 0.01)):
        for seed in [1, 2, 3, 4, 5]:
            params = (
                Parameters.from_base(mesh_size=3, seed=seed, sampling_p=sampling_p)
                .multi_resolution(1, p_sched=[5, 5, 5])
                .gomea(fos=-6, partial_evals=True)
                .instance(Collection.EMPIRE, instance)
                .stopping_criteria(iterations=[1000])
            )
            yield Experiment(params, project)


def full_eval(instance: int, project: str):
    for seed in [i for i in range(1, 11)]:
        params = (
            Parameters.from_base(mesh_size=3, seed=seed, sampler="Full")
            .multi_resolution(1, p_sched=[5, 5, 5])
            .gomea(partial_evals=True, fos=-6)
            .instance(Collection.EMPIRE, instance)
            .stopping_criteria(iterations=[1000])
        )
        yield Experiment(params, project)


def pd_pop_experiment(project):
    for pop_size in [2**x for x in range(4, 11)]:
        for seed in [1, 2, 3, 4, 5]:
            params = (
                Parameters.from_base(mesh_size=2, sampling_p=0.05, seed=seed)
                .multi_resolution(1, p_sched=[5, 5, 5])
                .gomea(pop_size=pop_size)
                .instance(Collection.EMPIRE, 16)
                .shrinkage(False)
                .stopping_criteria(iterations=100)
            )
            yield Experiment(params, project)


def convergence_experiment(project):
    for instance in [16, 17, 14]:
        for seed in [1, 2, 3, 4, 5]:
            sched = [7, 7, 7] if instance == 14 else [5, 5, 5]
            params = (
                Parameters.from_base(mesh_size=3, sampling_p=0.1, seed=seed)
                .multi_resolution(1, sched)
                .gomea(partial_evals=True, fos=-6)
                .stopping_criteria(50)
                .instance(Collection.EMPIRE, instance)
            )
            yield Experiment(params, project)


def pareto_experiment(project, instance):
    for _ in range(100):
        weight0 = 1
        weight1 = random.uniform(1, 5000)
        params = (
            Parameters.from_base(mesh_size=2, sampling_p=0.05)
            .multi_resolution(1, p_sched=[7, 7, 7])
            .multi_metric(weight0=weight0, weight1=weight1)
            .asgd()
            .instance(Collection.EMPIRE, instance)
            .stopping_criteria(iterations=[3000])
        )
        yield Experiment(params, project)


if __name__ == "__main__":
    queue = ExperimentQueue()
    for experiment in full_eval(16, "16_fulleval"):
        queue.push(experiment)
