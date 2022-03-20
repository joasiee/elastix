import logging
import numpy as np

from elastix_wrapper.parameters import Collection, Parameters
from experiments.experiment import Experiment, ExperimentQueue

logger = logging.getLogger("ParetoFront")
expqueue = ExperimentQueue()

instances = [16, 23, 17, 13, 6]

def wandb_test():
    project = "wandb_test"
    params = Parameters.from_base(sampler="RandomCoordinate", sampling_p=0.2).gomea().stopping_criteria(iterations=100).instance(Collection.EXAMPLES, 1)
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
                  .multi_resolution(n=3, p_sched=sched)
                  .stopping_criteria(iterations=iterations)
                  .instance(Collection.EMPIRE, instance))

        experiment = Experiment(project, params.params)
        if experiment.already_done():
            continue

        expqueue.push(experiment)


def pareto_front(instance, gomea, n=200):
    mesh_size = 8
    sched = [5, 5, 5, 3, 3, 3]
    iterations_g = [50, 150]
    iterations_a = [100000, 300000]
    project = "pareto_front"
    optimizer = "GOMEA" if gomea else "AdaptiveStochasticGradientDescent"

    for _ in range(n):
        weight0 = np.around(np.random.uniform(0.001, 0.101), 3)
        weight1 = np.around(np.random.uniform(0.001, 1.001), 2)

        params = (Parameters.from_base(mesh_size=mesh_size)
                    .multi_metric(weight0=weight0, weight1=weight1)
                    .multi_resolution(n=2, p_sched=sched)
                    .instance(Collection.EMPIRE, instance)
                    )
        if gomea:
                params.gomea(
                    fos=-6, partial_evals=True).stopping_criteria(iterations=iterations_g)
        else:
            params.optimizer(optimizer).stopping_criteria(
                iterations=iterations_a)

        experiment = Experiment(project, params.params)
        if experiment.already_done(["Metric0Weight", "Metric1Weight"]):
            logger.info(f"{weight0}-{weight1} exists, not submitting as job")
            continue

        expqueue.push(experiment)


if __name__ == "__main__":
    wandb_test()