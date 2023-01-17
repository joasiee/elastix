from pathlib import Path
import wandb
import os
from thesispy.definitions import ROOT_DIR, Collection
import thesispy.elastix_wrapper.wrapper as wrapper
from thesispy.experiments.dataset import Dataset, FinishedRun
import pandas as pd

DOWNLOAD_FOLDER = ROOT_DIR / "output" / "download"

entity = "joasiee"
api = wandb.Api()


def download_file(out_dir: Path, filename: str, run):
    out_file = out_dir / filename
    file = run.file(filename)
    wandb.util.download_file_from_url(str(out_file.resolve()), file.url, os.environ["WANDB_API_KEY"])


def get_run_result(project: str, run_id: str):
    run = api.run(f"{entity}/{project}/{run_id}")
    out_dir = DOWNLOAD_FOLDER / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    download_file(out_dir, "TransformParameters.0.txt", run)
    download_file(out_dir, "controlpoints.dat", run)
    collection = Collection(run.config["Collection"])
    instance = int(run.config["Instance"])
    return wrapper.get_run_result(collection, instance, out_dir / "TransformParameters.0.txt")


def get_runs(project: str, filters: dict = None):
    return api.runs(f"{entity}/{project}", filters=filters)


def parse_run(run):
    return FinishedRun(run.name, run.config, pd.DataFrame.from_dict(run.scan_history()))


def get_runs_as_dataset(project, filters={}):
    local_ds = Dataset.load(project)
    if local_ds:
        return local_ds

    runs = []
    for run in api.runs(entity + "/" + project, filters=filters):
        if run.state == "finished":
            runs.append(parse_run(run))
    return Dataset(project, runs)
