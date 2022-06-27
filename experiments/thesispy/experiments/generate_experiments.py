import numpy as np
from thesispy.elastix_wrapper.parameters import Parameters, Collection
from thesispy.experiments.experiment import Experiment, ExperimentQueue


def yield_experiments(collection: Collection, instance: int, project: str, exp_fn):
    for params in exp_fn():
        params.instance(collection, instance)
        yield Experiment(params, project)


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


def gridsizes():
    for gridsize in range(2, 11):
        for seed in range(5):
            params = (
                Parameters.from_base(mesh_size=gridsize, sampler="Full", seed=seed)
                .multi_resolution(1, [5, 5, 5])
                .gomea()
                .stopping_criteria(20)
            )
            yield params


def tre_divergence():
    for gridsize in [2, 5, 7]:
        params = (
            Parameters.from_base(mesh_size=gridsize, sampler="Full", seed=1)
            .multi_resolution(1, [4, 4, 4])
            .multi_metric(
                metric0="AdvancedNormalizedCorrelation",
                metric1="CorrespondingPointsEuclideanDistanceMetric",
                weight1=0.0,
            )
            .asgd()
            .stopping_criteria(iterations=25000)
        )
        yield params


def regularization_weight():
    for weight in np.arange(200, 10200, 200):
        weight = int(weight)
        # weight = np.round(weight, 3)
        params = (
            Parameters.from_base(mesh_size=5, seed=1)
            .multi_resolution(1, [4, 4, 4])
            .multi_metric(metric1="TransformBendingEnergyPenalty", weight1=weight)
            .asgd()
            .stopping_criteria(iterations=10000)
        )
        yield params


def multi_resolution_settings():
    for its in [100, 1000, 10000, 25000, 50000]:
        params = (
            Parameters.from_base(mesh_size=5, seed=1)
            .multi_resolution(3)
            .multi_metric(metric1="TransformBendingEnergyPenalty", weight1=0.005)
            .asgd()
            .stopping_criteria(iterations=its)
        )
        yield params


def fos_settings():
    for setting in [-1, -6, 1]:
        peval = True if setting != -1 else False
        for seed in range(10):
            params = (
                Parameters.from_base(mesh_size=5, seed=seed)
                .multi_resolution(1, [5, 5, 5])
                .gomea(fos=setting, partial_evals=peval)
                .stopping_criteria(iterations=300)
            )
            yield params


if __name__ == "__main__":
    queue = ExperimentQueue()
    fn = fos_settings

    for experiment in yield_experiments(Collection.EMPIRE, 16, fn.__name__, fn):
        queue.push(experiment)
