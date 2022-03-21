import logging
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
    expqueue.push(experiment)


def convergence_tests():
    project = "convergence_tests"
    iterations = [50, 50, 150]
    sched = [5, 5, 5, 4, 4, 4, 3, 3, 3]

    for instance in instances:
        params = (Parameters.from_base(mesh_size=8)
                  .gomea(fos=-6, partial_evals=True)
                  .multi_metric()
                  .multi_resolution(n=3, p_sched=sched, g_sched=sched)
                  .stopping_criteria(iterations=iterations)
                  .instance(Collection.EMPIRE, instance))

        experiment = Experiment(project, params.params)
        if experiment.already_done():
            continue

        expqueue.push(experiment)


def pareto_front(instance: int, gomea: bool, n: int = 200):
    sched = [6, 6, 6, 5, 5, 5, 4, 4, 4]
    iterations_g = [100, 50, 30]
    iterations_a = [1000, 500, 300]
    project = "pareto_front_sampling"
    rseed = 12378842

    for _ in range(n):
        weight0 = np.around(np.random.uniform(0.001, 0.101), 3)
        weight1 = np.around(np.random.uniform(0.001, 1.001), 2)

        params = (Parameters.from_base(mesh_size=8, seed=rseed)
                  .multi_resolution(3, p_sched=sched)
                  .multi_metric(weight0=weight0, weight1=weight1)
                  .instance(Collection.EMPIRE, instance))

        if gomea:
            params.gomea(
                fos=-6, partial_evals=True).stopping_criteria(iterations=iterations_g)
        else:
            params.optimizer("AdaptiveStochasticGradientDescent").stopping_criteria(
                iterations=iterations_a)

        experiment = Experiment(project, params)
        expqueue.push(experiment)


if __name__ == "__main__":
    pareto_front(7, False, 200)