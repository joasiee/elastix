from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import wandb
import elastix_wrapper.wrapper as wrapper
from elastix_wrapper.parameters import Collection, Parameters

for instance in range(1,31):
    params = Parameters().gomea(fos=-6, partial_evals=True).stopping_criteria(iterations=50).instance(Collection.EMPIRE, instance)
    run = wandb.init(project="convergence_tests", name=str(params), reinit=True)
    wandb.config.instance = instance
    wrapper.run(params, Path("output") / wandb.run.project / wandb.run.name)
    run.finish()