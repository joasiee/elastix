import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from elastix_wrapper.parameters import Collection, Parameters
import elastix_wrapper.wrapper as wrapper
import wandb
import numpy as np
import logging

logger = logging.getLogger("ParetoFront")

GOMEA = bool(int(sys.argv[1]))
instance = int(sys.argv[2])
entity, project = "joasiee", "pareto_front_sampling"

sched = [6, 6, 6, 5, 5, 5, 4, 4, 4]
iterations_g = [100, 50, 30]
iterations_a = [3000, 2000, 1000]
project = "pareto_front_sampling"
rseed = 12378842

for _ in range(20):
    weight0 = np.around(np.random.uniform(0.001, 0.101), 3)
    weight1 = np.around(np.random.uniform(0.001, 1.001), 2)

    params = (Parameters.from_base(mesh_size=8, seed=rseed)
                .multi_resolution(3, p_sched=sched)
                .multi_metric(weight0=weight0, weight1=weight1)
                .instance(Collection.EMPIRE, instance))

    if GOMEA:
        params.gomea(
            fos=-6, partial_evals=True).stopping_criteria(iterations=iterations_g)
    else:
        params.optimizer("AdaptiveStochasticGradientDescent").stopping_criteria(
            iterations=iterations_a)

    run = wandb.init(project=project,
                        name=str(params), reinit=True)
    params.prune()
    wandb.config.update(params.params)
    wrapper.run(params, Path("output") / wandb.run.project / wandb.run.name)