import itertools
import numpy as np
from thesispy.elastix_wrapper.parameters import Parameters
from thesispy.experiments.experiment import Experiment, ExperimentQueue
from thesispy.definitions import *


def yield_experiments(collection: Collection, instance: int, project: str, exp_fn):
    for params in exp_fn():
        params.instance(collection, instance)
        project_name = f"{collection.name.lower()}_{instance}_{project}"
        yield Experiment(params, project_name)


def regularization_weight():
    for seed in range(5):
        seed += 1
        for weight in [0.0] + list(np.geomspace(0.0001, 10.0, 30)):
            weight = np.round(weight, 5)
            params = (
                Parameters.from_base(mesh_size=6, seed=seed)
                .asgd()
                .regularize(weight)
                .stopping_criteria(iterations=10000)
            )
            yield params

            params = (
                Parameters.from_base(mesh_size=6, seed=seed)
                .gomea(LinkageType.CP_MARGINAL, hybrid=True)
                .regularize(weight)
                .stopping_criteria(iterations=300)
            )
            yield params

            params = (
                Parameters.from_base(mesh_size=6, seed=seed)
                .gomea(LinkageType.CP_MARGINAL, hybrid=False)
                .regularize(weight)
                .stopping_criteria(iterations=300)
            )
            yield params


def regularization_weight_subsampling():
    for weight in [0.0] + list(np.geomspace(0.0001, 10.0, 50)):
        weight = np.round(weight, 4)
        params = (
            Parameters.from_base(mesh_size=5)
            .asgd()
            .sampler("Random", pct=0.1)
            .regularize(weight)
            .stopping_criteria(iterations=5000)
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


def fair_comparison_hybrid():
    for seed in range(10):
        seed += 1
        for mesh_size in [3, 4, 5]:
            params = (
                Parameters.from_base(mesh_size=mesh_size, seed=seed)
                .gomea(LinkageType.CP_MARGINAL, hybrid=True)
                .stopping_criteria(iterations=1000)
            )
            yield params

            params = (
                Parameters.from_base(mesh_size=mesh_size, seed=seed)
                .asgd()
                .stopping_criteria(iterations=10000)
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
    for seed in range(10):
        seed += 1
        params = (
            Parameters.from_base(mesh_size=4, seed=seed)
            .gomea(LinkageType.CP_MARGINAL, use_constraints=True, compute_folds_constraints=True)
            .stopping_criteria(iterations=1000)
        )
        yield params


def fold_constraints():
    for seed in range(10):
        for mesh_size in [4, 5, 6]:
            param_folds = (
                Parameters.from_base(mesh_size=mesh_size, seed=seed)
                .gomea(
                    LinkageType.CP_MARGINAL, use_constraints=True, compute_folds_constraints=True
                )
                .stopping_criteria(iterations=1000)
            )
            yield param_folds
            param_no_folds = (
                Parameters.from_base(mesh_size=mesh_size, seed=seed)
                .gomea(
                    LinkageType.CP_MARGINAL, use_constraints=False, compute_folds_constraints=False
                )
                .stopping_criteria(iterations=1000)
            )
            yield param_no_folds


def linkage_models():
    peval_budget = 30000e6
    max_iterations = 10000
    for seed in range(10):
        seed += 1
        for linkage in [
            LinkageType.UNIVARIATE,
            LinkageType.CP_MARGINAL,
            LinkageType.STATIC_EUCLIDEAN,
            LinkageType.FULL,
        ]:
            for mesh_size in [5, 7, 9]:
                params = (
                    Parameters.from_base(mesh_size=mesh_size, seed=seed)
                    .gomea(linkage, min_set_size=6, max_set_size=9)
                    .stopping_criteria(iterations=max_iterations, pixel_evals=peval_budget)
                )
                yield params


def hybrid_sweep():
    for tau_asgd in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        for iterations_asgd in [5, 10, 20, 30, 50, 100, 200, 500]:
            params = (
                Parameters.from_base(mesh_size=5, seed=88)
                .gomea(
                    LinkageType.CP_MARGINAL,
                    hybrid=True,
                    tau_asgd=tau_asgd,
                    asgd_iterations=iterations_asgd,
                    redis_method=RedistributionMethod.Random,
                    it_schedule=IterationSchedule.Static,
                )
                .stopping_criteria(iterations=5000, pixel_evals=50000e6)
            )
            yield params


def asgd_sweep():
    for nr_resolutions in [1, 2, 3, 4, 5]:
        p_sched = [i for i in range(nr_resolutions + 1, 1, -1)]
        for mesh_size in [5, 6, 8, 10, 12]:
            params = (
                Parameters.from_base(
                    mesh_size=mesh_size,
                    seed=83,
                    metric="AdvancedNormalizedCorrelation",
                    use_mask=True,
                )
                .asgd()
                .multi_resolution(
                    nr_resolutions, p_sched=p_sched, g_sched=[1 for _ in range(nr_resolutions)]
                )
                .stopping_criteria(iterations=2000)
            )
            yield params


def fair_comparison_final():
    for seed in range(2):
        seed += 1
        for mesh_size in [6, 9]:
            for reg_weight in [0.0001, 0.001, 0.01, 0.1]:
                peval_budget = 300000e6
                base = (
                    lambda: Parameters.from_base(
                        mesh_size=mesh_size,
                        seed=seed,
                        metric="AdvancedNormalizedCorrelation",
                        use_mask=True,
                    )
                    .regularize(reg_weight)
                    .multi_resolution(3, r_sched=[5, 4, 3], s_sched=[6, 2, 0], g_sched=[2, 2, 1])
                )

                params_gomea = (
                    base()
                    .gomea(LinkageType.CP_MARGINAL)
                    .stopping_criteria(iterations=[200, 200, 2000], pixel_evals=peval_budget)
                )
                yield params_gomea

                params_gomea_ls = (
                    base()
                    .gomea(
                        LinkageType.CP_MARGINAL,
                        hybrid=True,
                        redis_method=RedistributionMethod.BestN,
                        it_schedule=IterationSchedule.Logarithmic,
                    )
                    .stopping_criteria(iterations=[200, 200, 2000], pixel_evals=peval_budget)
                )
                yield params_gomea_ls

                params_gomea_fc = (
                    base()
                    .gomea(
                        LinkageType.CP_MARGINAL,
                        use_constraints=True,
                        compute_folds_constraints=True,
                    )
                    .stopping_criteria(iterations=[200, 200, 2000], pixel_evals=peval_budget)
                )
                yield params_gomea_fc

                params_asgd = base().asgd().stopping_criteria(iterations=[500, 500, 4000])
                yield params_asgd


if __name__ == "__main__":
    queue = ExperimentQueue()
    queue.clear()
    fn = fair_comparison_final

    # Collection + instance niet vergeten!
    queue.bulk_push(list(yield_experiments(Collection.LEARN, 1, fn.__name__, fn)))
    print(f"Queue size: {queue.size()}")
