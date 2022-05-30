import random

import numpy as np
from thesispy.elastix_wrapper.parameters import Parameters, Collection
from thesispy.experiments.experiment import Experiment, ExperimentQueue


def yield_experiments(collection: Collection, instance: int, project: str, exp_fn):
    for params in exp_fn():
        params.instance(collection, instance)
        yield Experiment(params, project)


def sampling_p_range():
    for sampling_p in list(np.arange(0.25, 0.81, 0.05)):
        for seed in [1, 2, 3]:
            params = (
                Parameters.from_base(mesh_size=3, seed=seed, sampling_p=sampling_p)
                .multi_resolution(1, p_sched=[5, 5, 5])
                .gomea(fos=-6, partial_evals=True)
                .stopping_criteria(iterations=[500])
            )
            yield params


def full_eval():
    for seed in [i for i in range(1, 6)]:
        params = (
            Parameters.from_base(mesh_size=5, seed=seed, sampler="Full")
            .multi_resolution(1, p_sched=[4, 4, 4])
            .asgd()
            .stopping_criteria(iterations=[1000])
        )
        yield params


def pd_pop_experiment():
    for pop_size in [2**x for x in range(4, 11)]:
        for seed in [1, 2, 3, 4, 5]:
            params = (
                Parameters.from_base(mesh_size=2, sampling_p=0.05, seed=seed)
                .multi_resolution(1, p_sched=[5, 5, 5])
                .gomea(pop_size=pop_size)
                .stopping_criteria(iterations=100)
            )
            yield params


def convergence_experiment():
    for instance in [16, 17, 14]:
        for seed in [1, 2, 3, 4, 5]:
            sched = [7, 7, 7] if instance == 14 else [5, 5, 5]
            params = (
                Parameters.from_base(mesh_size=3, sampling_p=0.1, seed=seed)
                .multi_resolution(1, sched)
                .gomea(partial_evals=True, fos=-6)
                .stopping_criteria(50)
            )
            yield params


def pareto_experiment():
    for _ in range(100):
        weight0 = 1
        weight1 = int(random.uniform(1, 1000.5))
        params = (
            Parameters.from_base(mesh_size=4, sampling_p=0.05)
            .multi_resolution(1, p_sched=[7, 7, 7])
            .multi_metric(weight0=weight0, weight1=weight1)
            .asgd()
            .stopping_criteria(iterations=[3000])
        )
        yield params


def grid_experiment():
    for gridsize in [2, 3, 4, 5, 6, 7]:
        params = (
            Parameters.from_base(mesh_size=5, sampler="Full", seed=1)
            .multi_resolution(1, [5, 5, 5])
            .gomea(partial_evals=True, fos=-6)
            .stopping_criteria(500)
        )
        yield params

def tre_divergence():
    for gridsize in [2, 5, 7]:
        params = (
            Parameters.from_base(mesh_size=gridsize, sampler="Full", seed=1)
            .multi_resolution(1, [4,4,4])
            .multi_metric(metric0="AdvancedNormalizedCorrelation", metric1="CorrespondingPointsEuclideanDistanceMetric", weight1=0.0)
            .asgd()
            .stopping_criteria(iterations=25000)
        )
        yield params


if __name__ == "__main__":
    queue = ExperimentQueue()
    for experiment in yield_experiments(Collection.LEARN, 1, "tre_divergence_learn", tre_divergence):
        queue.push(experiment)