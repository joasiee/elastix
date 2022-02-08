from pathlib import Path
from typing import Dict
import experiments.wrapper as wrapper
from experiments.db import DBClient
from experiments.parameters import Parameters, Collection
from enum import Enum
import nibabel as nib

class EXPERIMENT(Enum):
    PARETO_APPROXIMATION = 1


BASE_PARAMS: Dict[EXPERIMENT, Parameters] = {
    EXPERIMENT.PARETO_APPROXIMATION: (Parameters(mesh_size=8).
                                       gomea(fos=-6, partial_evals=True)
                                       .multi_metric(weight1=100)
                                       .stopping_criteria(iterations=5000)
                                       .instance(Collection.EMPIRE, 1))
}

def run_experiment(experiment: EXPERIMENT, params: Parameters):
    db = DBClient(experiment)
    db.save_results(wrapper.run(params))

if __name__ == "__main__":
    # experiment = EXPERIMENT.PARETO_APPROXIMATION
    # params = BASE_PARAMS[experiment]
    # params.write(Path())

    # run_experiment(experiment.name, params)
    print(nib.load(Path("case_001_exp.nii")).shape)