import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from elastix_wrapper.parameters import Collection, Parameters
import elastix_wrapper.wrapper as wrapper
import wandb


instances = [16, 23, 17, 13, 6]
downsampling_f = 5

for instance in instances:
    params = Parameters(mesh_size=8, downsampling_f=downsampling_f).gomea(
        fos=-6, partial_evals=True).stopping_criteria(iterations=100).instance(Collection.EMPIRE, instance)
    run = wandb.init(project="convergence_tests",
                     name=str(params), reinit=True)
    wandb.config.instance = instance
    wandb.config.downsampling = downsampling_f
    wrapper.run(params, Path("output") / wandb.run.project / wandb.run.name)
    run.finish()
