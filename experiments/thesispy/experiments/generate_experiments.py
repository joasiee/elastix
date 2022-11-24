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
    for weight in np.geomspace(0.01, 10.0, 20):
        weight = np.round(weight, 3)
        params = (
            Parameters.from_base(mesh_size=4)
            .gomea(LinkageType.CP_MARGINAL, use_constraints=True, contraints_threshold=0.005)
            .regularize(weight)
            .stopping_criteria(iterations=500)
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
    peval_budget = 100000e6
    max_iterations = 10000
    for seed in range(10):
        seed += 1
        for linkage in [
            # LinkageType.UNIVARIATE,
            # LinkageType.CP_MARGINAL,
            LinkageType.STATIC_EUCLIDEAN,
            # LinkageType.FULL,
        ]:
            params = (
                Parameters.from_base(mesh_size=5, seed=seed, use_missedpixel_penalty=True)
                .gomea(linkage, min_set_size=3, max_set_size=6)
                .stopping_criteria(iterations=max_iterations, pixel_evals=peval_budget)
            )
            yield params


if __name__ == "__main__":
    queue = ExperimentQueue()
    queue.clear()
    fn = regularization_weight

    queue.bulk_push(list(yield_experiments(Collection.SYNTHETIC, 1, fn.__name__, fn)))
    print(f"Queue size: {queue.size()}")
