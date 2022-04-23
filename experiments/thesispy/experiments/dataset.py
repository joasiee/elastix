from pathlib import Path
from typing import Any, Dict, List
import itertools
import pickle
import numpy as np
import pandas as pd
import dictquery as dq

from thesispy.definitions import ROOT_DIR

DATASETS_PATH = ROOT_DIR / Path("datasets")
if not DATASETS_PATH.exists():
    DATASETS_PATH.mkdir(parents=True)


class FinishedRun:
    def __init__(
        self, name: str, config: Dict[str, Any], metrics: pd.DataFrame
    ) -> None:
        self.name = name
        self.config = config
        self.resolutions_train = []
        self.resolutions_val = []

        nr_resolutions = int(self.config["NumberOfResolutions"])
        for r in range(0, nr_resolutions):
            condition = (
                ~np.isnan(metrics[f"R{r}/metric"])
                if nr_resolutions > 1
                else metrics.index
            )
            indices = metrics.index[condition]
            columns = ["_step", "_runtime", "_timestamp"] + [
                c for c in metrics.columns if f"R{r}/" in c
            ]
            metrics_r = metrics[columns]
            metrics_r.columns = [c.replace(f"R{r}/", "") for c in metrics_r.columns]
            self.resolutions_train.append(metrics_r.loc[indices].iloc[:-1])
            self.resolutions_val.append(metrics_r.loc[indices].iloc[-1])

    def query(self, query: str):
        return dq.match(self.config, query)


class Dataset:
    def __init__(self, project: str, runs: List[FinishedRun]) -> None:
        self.runs: List[FinishedRun] = runs
        self.project = project

    def add_run(self, run: FinishedRun):
        self.runs.append(run)

    def filter(self, query: str):
        return Dataset(self.project, [run for run in self.runs if run.query(query)])

    def groupby(self, attrs: List[str]):
        if len(attrs) == 0:
            yield (), self.runs
        else:
            unique_values = [set() for _ in range(len(attrs))]
            for i, attr in enumerate(attrs):
                for run in self.runs:
                    unique_values[i].add(run.config[attr])

            for unique_value_tuple in itertools.product(*unique_values):
                query = ""
                for i, unique_value in enumerate(unique_value_tuple):
                    query += f"{attrs[i]} == {unique_value} AND "
                query = query[:-5]
                yield unique_value_tuple, self.filter(query).runs

    def aggregate(
        self,
        attrs: List[str],
        metrics: List[str],
        resolution: int = 0,
        val: bool = True,
    ):
        df = pd.DataFrame(columns=metrics)
        for group, runs in self.groupby(attrs):
            df_add = pd.DataFrame(columns=metrics)
            for run in runs:
                if val:
                    val_df = run.resolutions_val[resolution][metrics].to_frame()
                    val_df = val_df.transpose()
                    df_add = pd.concat([df_add, val_df])
                else:
                    df_add = pd.concat(
                        [df_add, run.resolutions_train[resolution][metrics]]
                    )
            for i, attr in enumerate(attrs):
                df_add[attr] = group[i]
            df = pd.concat([df, df_add])
        return df

    def save(self):
        path = DATASETS_PATH / f"{self.project}.pkl"
        with path.open("wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(project: str):
        path = DATASETS_PATH / f"{project}.pkl"
        with path.open("rb") as file:
            return pickle.load(file)
