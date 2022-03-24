import logging

import numpy as np
from elastix_wrapper.parameters import Parameters, Collection
from experiments.experiment import Experiment, ExperimentQueue

logger = logging.getLogger("ParetoFront")

def sampling_p_range(instance: int, project: str):
    for sampling_p in np.arange(0.01, 0.21, 0.01):
        params = (
            Parameters.from_base(mesh_size=2, seed=1523, sampling_p=sampling_p)
            .multi_resolution(1, p_sched=[7, 7, 7])
            .multi_metric()
            .gomea()
            .instance(Collection.EMPIRE, 26)
            .stopping_criteria(iterations=[100])
        )
        yield Experiment(params, project)

if __name__ == "__main__":
    queue = ExperimentQueue()
    for experiment in sampling_p_range(26, "gomea_uni_sampling"):
        queue.push(experiment)
