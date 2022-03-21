import logging
import random
from typing import List
import numpy as np

from elastix_wrapper.parameters import Collection, Parameters
from experiments.experiment import Experiment, ExperimentQueue

logger = logging.getLogger("ParetoFront")
expqueue = ExperimentQueue()

instances = [1, 4, 7, 8, 10, 14, 15, 18, 20, 21, 28]


def wandb_test():
    project = "wandb_test"
    params = Parameters.from_base(sampler="RandomCoordinate", sampling_p=0.2).gomea(
    ).stopping_criteria(iterations=100).instance(Collection.EXAMPLES, 1)
    experiment = Experiment(project, params)
    return experiment


def convergence_tests(instance: int):
    project = "convergence_tests"
    iterations = [50, 50, 150]
    sched = [6, 6, 6, 5, 5, 5, 4, 4, 4]
    params = (Parameters.from_base(mesh_size=8)
              .gomea(fos=-6, partial_evals=True)
              .multi_metric()
              .multi_resolution(n=3, p_sched=sched, g_sched=sched)
              .stopping_criteria(iterations=iterations)
              .instance(Collection.EMPIRE, instance))

    experiment = Experiment(project, params)
    if experiment.already_done():
        return None
    return experiment


def pareto_front(instance: int, gomea: bool, reps: int = 5) -> List[Experiment]:
    sched = [7, 7, 7, 6, 6, 6, 5, 5, 5]
    iterations_g = [30, 50, 125]
    iterations_a = [2000, 2000, 3000]
    project = "pareto_front"
    seeds = [int((random.random() + 0.01) * 100000) for _ in range(reps)]

    weight0 = np.around(np.random.uniform(0.001, 0.201), 3)
    weight1 = np.around(np.random.uniform(0.001, 1.001), 2)

    experiments = []

    for seed in seeds:
        params = (Parameters.from_base(mesh_size=8, seed=seed)
                  .multi_resolution(3, p_sched=sched)
                  .multi_metric(weight0=weight0, weight1=weight1)
                  .instance(Collection.EMPIRE, instance))

        if gomea:
            params.gomea(
                fos=-6, partial_evals=True).stopping_criteria(iterations=iterations_g)
        else:
            params.optimizer("AdaptiveStochasticGradientDescent").stopping_criteria(
                iterations=iterations_a)

        experiments.append(Experiment(project, params))

    return experiments


if __name__ == "__main__":
    pareto_front(7, False)
