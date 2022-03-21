import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import elastix_wrapper.wrapper as wrapper
from experiments.experiment import ExperimentQueue
import wandb

exp_queue = ExperimentQueue()

while exp_queue.peek():
    experiment = exp_queue.pop()
    run = wandb.init(project=experiment.project,
                     name=str(experiment.params), reinit=True)
    wandb.config.update(experiment.params.params)
    wrapper.run(experiment.params, Path("output") / wandb.run.project / wandb.run.name)