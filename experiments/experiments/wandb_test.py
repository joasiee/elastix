from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import wandb
import elastix_wrapper.wrapper as wrapper
from elastix_wrapper.parameters import Collection, Parameters

params = Parameters(downsampling_f=1, sampler="Full").gomea().stopping_criteria(iterations=10).instance(Collection.EXAMPLES, 1)
run = wandb.init(project="wandb_test", name=str(params))
wrapper.run(params, Path("output") / wandb.run.project / wandb.run.name)
run.finish()