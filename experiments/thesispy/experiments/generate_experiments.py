from math import log2
import numpy as np
from thesispy.elastix_wrapper.parameters import GOMEAType, Parameters, Collection
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
    for gridsize in [12]:
        for seed in range(5):
            params = (
                Parameters.from_base(mesh_size=gridsize, seed=seed)
                .multi_resolution(1, [3, 3, 3])
                .gomea()
                .stopping_criteria(30)
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
    for setting in [
        # GOMEAType.GOMEA_CP,
        # GOMEAType.GOMEA_UNIVARIATE,
        GOMEAType.GOMEA_FULL,
    ]:
        for seed in range(5):
            params = (
                Parameters.from_base(mesh_size=4, seed=seed, metric="AdvancedMeanSquares")
                .gomea(setting, pop_size=1000)
                .stopping_criteria(iterations=300)
            )
            yield params


def pop_sizes_cp():
    pop_size_fn = lambda n, n_params: int(31.7 + n * log2(n_params))
    for meshsize in [2, 4, 6, 10, 16]:
        nr_params = (meshsize + 3) ** 3 * 3
        for n in [2**x for x in range(6 + 1)]:
            params = (
                Parameters.from_base(mesh_size=meshsize, seed=1)
                .multi_resolution(1, [4, 4, 4])
                .gomea(fos=GOMEAType.GOMEA_CP, pop_size=pop_size_fn(n, nr_params))
                .stopping_criteria(iterations=100)
            )
            yield params


def subsampling_percentage():
    for seed in range(10):
        for pct in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
            for gomea in [True, False]:
                sampler = "Full" if pct == 1.0 else "RandomCoordinate"
                params = (
                    Parameters.from_base(mesh_size=5, seed=seed)
                    .multi_resolution(1, [4])
                    .sampler(sampler, pct=pct)
                )
                if gomea:
                    params.gomea(fos=GOMEAType.GOMEA_CP).stopping_criteria(
                        iterations=300
                    )
                else:
                    params.asgd().stopping_criteria(iterations=15000)

                yield params

def pareto_front_test():
    for _ in range(100):
        weight1 = np.random.uniform(0.15, 1.0)
        weight1 = np.around(weight1, 9)
        params = Parameters.from_base(mesh_size=5, metric="AdvancedMeanSquares").regularize(weight1)
        params.gomea(GOMEAType.GOMEA_CP).stopping_criteria(iterations=50)

        yield params

def nomask_test():
    for _ in range(10):
        params = Parameters.from_base(mesh_size=5, metric="AdvancedMeanSquares", use_mask=False)
        params.gomea(GOMEAType.GOMEA_CP).stopping_criteria(100)
        yield params

        params = Parameters.from_base(mesh_size=5, metric="AdvancedMeanSquares", use_mask=False)
        params.asgd().stopping_criteria(5000)
        yield params

def queue_test():
    for _ in range(10):
        params = Parameters.from_base(mesh_size=5).asgd().stopping_criteria(iterations=100)
        yield params


if __name__ == "__main__":
    queue = ExperimentQueue()
    fn = pareto_front_test

    queue.bulk_push(list(yield_experiments(Collection.SYNTHETIC, 1, fn.__name__, fn)))
