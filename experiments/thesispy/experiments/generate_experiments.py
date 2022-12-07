import itertools
import numpy as np
from thesispy.elastix_wrapper.parameters import Parameters
from thesispy.experiments.experiment import Experiment, ExperimentQueue
from thesispy.definitions import LinkageType, Collection


def yield_experiments(collection: Collection, instance: int, project: str, exp_fn):
    for params in exp_fn():
        params.instance(collection, instance)
        project_name = f"{collection.name.lower()}_{instance}_{project}"
        yield Experiment(params, project_name)


def regularization_weight():
    for weight in [0.0] + list(np.geomspace(0.0001, 10.0, 50)):
        weight = np.round(weight, 5)
        params = (
            Parameters.from_base(mesh_size=5)
            .asgd()
            .regularize(weight)
            .stopping_criteria(iterations=10000)
        )
        yield params
        params = (
            Parameters.from_base(mesh_size=5)
            .gomea()
            .regularize(weight)
            .stopping_criteria(iterations=300)
        )
        yield params

def regularization_weight_subsampling():
    for weight in [0.0] + list(np.geomspace(0.001, 5.0, 49)):
        weight = np.round(weight, 4)
        params = (
            Parameters.from_base(mesh_size=5)
            .asgd()
            .sampler("Random", pct=0.05)
            .regularize(weight)
            .stopping_criteria(iterations=1000)
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
                    params.gomea(fos=LinkageType.CP_MARGINAL).stopping_criteria(iterations=300)
                else:
                    params.asgd().stopping_criteria(iterations=15000)

                yield params


def nomask_msd():
    mesh_sizes = [2, 4, 6, 10]
    for seed in range(3):
        seed += 1
        for mesh_size in mesh_sizes:
            params = (
                Parameters.from_base(
                    mesh_size=mesh_size, metric="AdvancedMeanSquares", seed=seed, use_mask=False
                )
                .gomea(LinkageType.CP_MARGINAL)
                .stopping_criteria(iterations=500)
            )
            yield params


def fair_comparison():
    for seed in range(10):
        seed += 1
        for mesh_size in [3, 4, 5]:
            for optimizer in ["ASGD", "GOMEA"]:
                if optimizer == "ASGD":
                    params = (
                        Parameters.from_base(
                            mesh_size=mesh_size,
                            metric="AdvancedMeanSquares",
                            seed=seed,
                            use_mask=False,
                        )
                        .asgd()
                        .stopping_criteria(iterations=50000)
                    )
                else:
                    params = (
                        Parameters.from_base(
                            mesh_size=mesh_size,
                            metric="AdvancedMeanSquares",
                            seed=seed,
                            use_mask=False,
                        )
                        .gomea(LinkageType.CP_MARGINAL, constraints=False)
                        .stopping_criteria(iterations=2000)
                    )
                yield params


def fair_comparison_multiresolution():
    for seed in range(10):
        seed += 1
        for mesh_size in [3, 4, 5]:
            for optimizer in ["ASGD"]:
                if optimizer == "ASGD":
                    params = (
                        Parameters.from_base(
                            mesh_size=mesh_size,
                            metric="AdvancedMeanSquares",
                            seed=seed,
                            use_mask=False,
                        )
                        .asgd()
                        .multi_resolution(3, g_sched=[1, 1, 1], downsampling=False)
                        .stopping_criteria(iterations=[5000, 8000, 50000])
                    )
                else:
                    params = (
                        Parameters.from_base(
                            mesh_size=mesh_size,
                            metric="AdvancedMeanSquares",
                            seed=seed,
                            use_mask=False,
                        )
                        .gomea(LinkageType.CP_MARGINAL, constraints=False)
                        .multi_resolution(3, g_sched=[1, 1, 1], downsampling=False)
                        .stopping_criteria(iterations=[50, 50, 500])
                    )
                yield params


def constrained_selection():
    for seed in range(10):
        seed += 1
        for constraint_threshold in [0.0, 0.005, 0.02, 0.05, 0.1]:
            params_constrained = (
                Parameters.from_base(mesh_size=4, metric="AdvancedMeanSquares", seed=seed)
                .gomea(
                    LinkageType.CP_MARGINAL,
                    use_constraints=True,
                    contraints_threshold=constraint_threshold,
                )
                .stopping_criteria(iterations=1000)
            )
            yield params_constrained
        params_penalty = (
            Parameters.from_base(
                mesh_size=4, metric="AdvancedMeanSquares", seed=seed, use_missedpixel_penalty=True
            )
            .gomea(LinkageType.CP_MARGINAL, use_constraints=False)
            .stopping_criteria(iterations=1000)
        )
        yield params_penalty


def linkage_models():
    peval_budget = 30000e6
    max_iterations = 10000
    for seed in range(10):
        seed += 1
        for linkage in [
            # LinkageType.UNIVARIATE,
            # LinkageType.CP_MARGINAL,
            LinkageType.STATIC_EUCLIDEAN,
            # LinkageType.FULL,
        ]:
            for mesh_size in [5, 7, 9]:
                params = (
                    Parameters.from_base(mesh_size=mesh_size, seed=seed)
                    .gomea(linkage, min_set_size=6, max_set_size=9)
                    .stopping_criteria(iterations=max_iterations, pixel_evals=peval_budget)
                )
                yield params


def linkage_models_static():
    max_iterations = 300
    for seed in range(3):
        seed += 1
        for min, max in itertools.product([3, 6, 9, 12], [6, 9, 12, 15, 18, 24, 32]):
            if min >= max:
                continue
            params = (
                Parameters.from_base(mesh_size=5, seed=seed)
                .gomea(LinkageType.STATIC_EUCLIDEAN, min_set_size=min, max_set_size=max)
                .stopping_criteria(iterations=max_iterations)
            )
            yield params


if __name__ == "__main__":
    queue = ExperimentQueue()
    queue.clear()
    fn = linkage_models_static

    queue.bulk_push(list(yield_experiments(Collection.SYNTHETIC, 1, fn.__name__, fn)))
    print(f"Queue size: {queue.size()}")
