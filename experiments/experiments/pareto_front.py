import sys
from pathlib import Path
import wandb
import logging

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import elastix_wrapper.wrapper as wrapper
from experiments.generate_experiments import pareto_front

logger = logging.getLogger("ParetoFront")

GOMEA = bool(int(sys.argv[1]))
instance = int(sys.argv[2])
jobs_per = 20


for _ in range(jobs_per):
    for experiment in pareto_front(instance, GOMEA):
        params = experiment.params
        run = wandb.init(project=experiment.project,
                        name=str(params), reinit=True)
        params.prune()
        wandb.config.update(params.params)
        wrapper.run(params, Path("output") / wandb.run.project / wandb.run.name)
