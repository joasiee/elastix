import logging

import numpy as np
from elastix_wrapper.parameters import Parameters, Collection
from experiments.experiment import Experiment, ExperimentQueue

logger = logging.getLogger("ParetoFront")

def sampling_p_range(instance: int, project: str):
    for sampling_p in [0.01, 0.02, 0.03] + list(np.arange(0.05, 0.55, 0.05)):
        for seed in [2356, 3487, 4942, 1432]:
            params = (
                Parameters.from_base(mesh_size=2, seed=seed, sampling_p=sampling_p)
                .multi_resolution(1, p_sched=[7, 7, 7])
                .multi_metric()
                .gomea(fos=-6, partial_evals=True)
                .instance(Collection.EMPIRE, 26)
                .stopping_criteria(iterations=[200])
            )
            yield Experiment(params, project)

if __name__ == "__main__":
    queue = ExperimentQueue()
    for experiment in sampling_p_range(26, "gomea_sampling"):
        queue.push(experiment)
