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

mesh_size = 8
instances = [16, 23, 17, 13, 6] if len(sys.argv) == 2 else [int(sys.argv[2])]
sched = [5, 5, 5, 3, 3, 3]
iterations_g = [33, 100]
iterations_a = [50000, 100000]

GOMEA = bool(int(sys.argv[1]))
optimizer = "GOMEA" if GOMEA else "AdaptiveStochasticGradientDescent"
entity, project = "joasiee", "pareto_front"

for instance in instances:
    for weight0 in np.arange(0.01, 0.11, 0.01):
        for weight1 in np.arange(0.05, 1.05, 0.05):
            weight0 = np.around(weight0, 2)
            weight1 = np.around(weight1, 2)

            #check if already done
            filters = {"state": {"$in": ["finished", "running"]}, "config.Metric0Weight": float(weight0), "config.Metric1Weight": float(weight1), "config.Optimizer": optimizer, "config.Instance": instance}
            api = wandb.Api()
            runs = api.runs(entity + "/" + project, filters=filters)
            if len(runs) > 0:
                logger.info(f"Skipping run with {filters}")
                continue

            params = (Parameters.from_base(mesh_size=mesh_size)
                    .multi_metric(weight0=weight0, weight1=weight1)
                    .multi_resolution(n=2, p_sched=sched)
                    .instance(Collection.EMPIRE, instance)
                    )
            if GOMEA:
                params.gomea(fos=-6, partial_evals=True).stopping_criteria(iterations=iterations_g)
            else:
                params.optimizer(optimizer).stopping_criteria(iterations=iterations_a)
            run = wandb.init(project=project,
                            name=str(params), reinit=True)
            params.prune()
            wandb.config.update(params.params)
            wrapper.run(params, Path("output") / wandb.run.project / wandb.run.name)