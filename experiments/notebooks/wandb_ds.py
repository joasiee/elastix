import wandb
import pandas as pd
from thesispy.experiments.dataset import FinishedRun, Dataset

api = wandb.Api(timeout=30)
entity = "joasiee"


def parse_run(run):
    return FinishedRun(run.name, run.config, pd.DataFrame.from_dict(run.scan_history()))


def get_runs_as_dataset(project, filters={}):
    local_ds = Dataset.load(project)
    if local_ds:
        return local_ds

    runs = []
    for run in api.runs(entity + "/" + project, filters=filters):
        runs.append(parse_run(run))
    return Dataset(project, runs)
