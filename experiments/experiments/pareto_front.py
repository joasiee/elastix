import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from elastix_wrapper.parameters import Collection, Parameters
import elastix_wrapper.wrapper as wrapper
import wandb
import numpy as np


mesh_size = 8
instances = [16, 23, 17, 13, 6] if len(sys.argv) == 1 else [int(sys.argv[2])]
downsampling_f = 4
sched = [1.5, 1.5, 1.5, 1, 1, 1]
iterations_g = [25, 80]
iterations_a = [400000, 10000000]
max_time = 5000

GOMEA = bool(int(sys.argv[1]))

for instance in instances:
    for weight0 in np.arange(0.1, 1.1, 0.1):
        for weight1 in np.arange(1, 11, 1):
            params = (Parameters(mesh_size=mesh_size, downsampling_f=downsampling_f)
                    .multi_metric(weight0=weight0, weight1=weight1)
                    .multi_resolution(n=2, p_sched=sched, g_sched=sched)
                    .instance(Collection.EMPIRE, instance)
                    )
            if GOMEA:
                params.gomea(fos=-6, partial_evals=True).stopping_criteria(max_time_s=max_time, iterations=iterations_g)
            else:
                params.optimizer("AdaptiveStochasticGradientDescent").stopping_criteria(max_time_s=max_time, iterations=iterations_a)
            run = wandb.init(project="pareto_front",
                            name=str(params), reinit=True)
            params.prune()
            wandb.config.update(params.params)
            wrapper.run(params, Path("output") / wandb.run.project / wandb.run.name)
            run.finish()
