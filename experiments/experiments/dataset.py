from typing import Any, Dict, List
import itertools

import numpy as np
import pandas as pd
import dictquery as dq

class FinishedRun:
    def __init__(self, name: str, config: Dict[str, Any], metrics: pd.DataFrame) -> None:
        self.name = name
        self.config = config
        self.resolutions_train = []
        self.resolutions_val = []
        
        nr_resolutions = int(self.config["NumberOfResolutions"])
        for r in range(0, nr_resolutions):
            condition = ~np.isnan(metrics[f"R{r}/metric"]) if nr_resolutions > 1 else metrics.index
            indices = metrics.index[condition]
            columns = ["_step", "_runtime", "_timestamp"] +  [c for c in metrics.columns if f"R{r}/" in c]
            metrics_r = metrics[columns]
            self.resolutions_train.append(metrics_r.loc[indices].iloc[:-1])
            self.resolutions_val.append(metrics_r.loc[indices].iloc[-1])

    def query(self, query: str):
        return dq.match(self.config, query)


class Dataset:
    def __init__(
        self, runs: List[FinishedRun]
    ) -> None:
        self.runs: List[FinishedRun] = runs

    def add_run(self, run: FinishedRun):
        self.runs.append(run)

    def filter(self, query: str):
        return [run for run in self.runs if run.query(query)]

    def groupby(self, attrs: List[str]):
        unique_values = [set() for _ in range(len(attrs))]
        for i, attr in enumerate(attrs):
            for run in self.runs:
                unique_values[i].add(run.config[attr])
        
        for unique_value_tuple in itertools.product(*unique_values):
            query = ""
            for i, unique_value in enumerate(unique_value_tuple):
                query += f"{attrs[i]} == {unique_value} AND "
            query = query[:-5]
            yield unique_value_tuple, self.filter(query)
