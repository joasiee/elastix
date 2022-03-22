import logging
from typing import List
import numpy as np

from elastix_wrapper.parameters import Collection, Parameters
from experiments.experiment import Experiment

logger = logging.getLogger("ParetoFront")
instances = [1, 4, 7, 8, 10, 14, 15, 18, 20, 21, 28]
seeds = [40800, 26630, 41915, 58715, 63988, 47809, 73594, 19112, 38953, 89687]

def wandb_test():
    project = "wandb_test"
    params = (
        Parameters.from_base(sampler="RandomCoordinate", sampling_p=0.2)
        .gomea()
        .stopping_criteria(iterations=100)
        .instance(Collection.EXAMPLES, 1)
    )
    experiment = Experiment(project, params)
    return experiment


def convergence_tests(instance: int):
    project = "convergence_tests"
    iterations = [50, 50, 150]
    sched = [6, 6, 6, 5, 5, 5, 4, 4, 4]
    params = (
        Parameters.from_base(mesh_size=8)
        .gomea(fos=-6, partial_evals=True)
        .multi_metric()
        .multi_resolution(n=3, p_sched=sched, g_sched=sched)
        .stopping_criteria(iterations=iterations)
        .instance(Collection.EMPIRE, instance)
    )

    experiment = Experiment(project, params)
    if not experiment.already_done():
        return experiment


def pareto_front(instance: int, gomea: bool, n: int, reps: int = 5) -> List[Experiment]:
    sched = [7, 7, 7, 6, 6, 6, 5, 5, 5]
    iterations_g = [30, 50, 125]
    iterations_a = [2000, 2000, 3000]
    project = "pareto_front"

    for _ in range(n):
        weight0 = np.around(np.random.uniform(0.001, 0.101), 2)
        weight1 = np.around(np.random.uniform(0.001, 1.01), 2)

        for i in range(reps):
            seed = seeds[i]
            params = (
                Parameters.from_base(mesh_size=8, seed=seed)
                .multi_resolution(3, p_sched=sched)
                .multi_metric(weight0=weight0, weight1=weight1)
                .instance(Collection.EMPIRE, instance)
            )

            if gomea:
                params.gomea(fos=-6, partial_evals=True).stopping_criteria(
                    iterations=iterations_g
                )
            else:
                params.optimizer("AdaptiveStochasticGradientDescent").stopping_criteria(
                    iterations=iterations_a
                )
            
            experiment = Experiment(project, params)
            yield experiment


if __name__ == "__main__":
    for exp in pareto_front(7, False, 5, 5):
        print(exp.params)
