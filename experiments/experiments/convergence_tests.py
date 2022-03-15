import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from elastix_wrapper.parameters import Collection, Parameters
import elastix_wrapper.wrapper as wrapper
import wandb


instances = [16, 23, 17, 13, 6]
downsampling_f = 4
iterations = [30, 50]

for instance in instances:
    params = Parameters(mesh_size=8, downsampling_f=downsampling_f).gomea(
        fos=-6, partial_evals=True).multi_metric().multi_resolution(n=2).stopping_criteria(iterations=iterations).instance(Collection.EMPIRE, instance)
    run = wandb.init(project="convergence_tests",
                     name=str(params), reinit=True)
    params.prune()
    wandb.config.update(params.params)
    wrapper.run(params, Path("output") / wandb.run.project / wandb.run.name)
    run.finish()
