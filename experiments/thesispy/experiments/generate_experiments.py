import numpy as np
from thesispy.elastix_wrapper.parameters import Parameters
from thesispy.experiments.experiment import Experiment, ExperimentQueue
from thesispy.definitions import LinkageType, Collection


def yield_experiments(collection: Collection, instance: int, project: str, exp_fn):
    for params in exp_fn():
        params.instance(collection, instance)
        project_name = f"{collection.name.lower()}_{instance}_{project}"
        yield Experiment(params, project_name)


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


def regularization_weight():
    for weight in np.arange(200, 10200, 200):
        weight = int(weight)
        # weight = np.round(weight, 3)
        params = (
            Parameters.from_base(mesh_size=5, seed=1)
            .multi_resolution(1, [4, 4, 4])
            .regularize(weight)
            .asgd()
            .stopping_criteria(iterations=10000)
        )
        yield params


def fos_settings_wmask_offset():
    for seed in range(1):
        seed += 1
        for setting in [
            LinkageType.UNIVARIATE,
            LinkageType.CP_MARGINAL,
            LinkageType.STATIC_EUCLIDEAN,
            LinkageType.FULL,
        ]:
            params = (
                Parameters.from_base(mesh_size=4, seed=seed, metric="AdvancedMeanSquares", use_mask=True)
                .gomea(setting)
                .stopping_criteria(iterations=500)
            )
            yield params

def subsampling_percentage():
    for seed in range(10):
        for pct in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
            for gomea in [True, False]:
                sampler = "Full" if pct == 1.0 else "RandomCoordinate"
                params = Parameters.from_base(mesh_size=5, seed=seed).multi_resolution(1, [4]).sampler(sampler, pct=pct)
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
                Parameters.from_base(mesh_size=mesh_size, metric="AdvancedMeanSquares", seed=seed, use_mask=False)
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
                            Parameters.from_base(mesh_size=mesh_size, metric="AdvancedMeanSquares", seed=seed, use_mask=False)
                            .asgd()
                            .stopping_criteria(iterations=50000)
                        )
                    else:
                        params = (
                            Parameters.from_base(mesh_size=mesh_size, metric="AdvancedMeanSquares", seed=seed, use_mask=False)
                            .gomea(LinkageType.CP_MARGINAL, constraints=False)
                            .stopping_criteria(iterations=2000)
                        )
                    yield params


def fair_comparison_multiresolution():
    for seed in range(10):
        seed += 1
        for mesh_size in [3, 4, 5]:
                for optimizer in ["ASGD", "GOMEA"]:
                    if optimizer == "ASGD":
                        params = (
                            Parameters.from_base(mesh_size=mesh_size, metric="AdvancedMeanSquares", seed=seed, use_mask=False)
                            .asgd()
                            .multi_resolution(3, g_sched=[1, 1, 1], downsampling=False)
                            .stopping_criteria(iterations=[2000, 5000, 20000])
                        )
                    else:
                        params = (
                            Parameters.from_base(mesh_size=mesh_size, metric="AdvancedMeanSquares", seed=seed, use_mask=False)
                            .gomea(LinkageType.CP_MARGINAL, constraints=False)
                            .multi_resolution(3, g_sched=[1, 1, 1], downsampling=False)
                            .stopping_criteria(iterations=[50, 50, 500])
                        )
                    yield params
        
    


if __name__ == "__main__":
    queue = ExperimentQueue()
    queue.clear()
    fn = fair_comparison_multiresolution

    queue.bulk_push(list(yield_experiments(Collection.SYNTHETIC, 1, fn.__name__, fn)))
    print(f"Queue size: {queue.size()}")
