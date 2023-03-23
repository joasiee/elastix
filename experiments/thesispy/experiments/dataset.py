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
    """A finished elastix run.
    
    Args:
        name: Name of the run.
        id: Unique id of the run.
        config: Dictionary of the configuration of the run.
        resolutions: List of dataframes containing the iteration output of the run for each resolution.
        summary: Series containing the final iteration output values.
    """

    def __init__(self, name: str, id: str, config: Dict[str, Any], metrics: pd.DataFrame) -> None:
        self.name = name
        self.id = id
        self.config = config
        metrics = metrics.astype({"_step": "uint64", "_runtime": "ulonglong", "_timestamp": "ulonglong"})
        self.resolutions = []
        self.summary = pd.Series(dtype="float64")

        nr_resolutions = int(self.config["NumberOfResolutions"])
        for r in range(0, nr_resolutions):
            condition = ~np.isnan(metrics[f"R{r}/time[ms]"])
            indices = metrics.index[condition]
            columns = ["_step", "_runtime", "_timestamp"] + [c for c in metrics.columns if f"R{r}/" in c]
            metrics_r = metrics[columns]
            metrics_r.columns = [c.replace(f"R{r}/", "") for c in metrics_r.columns]
            self.resolutions.append(metrics_r.loc[indices])

            if r == nr_resolutions - 1:
                self.summary = metrics_r.loc[indices].iloc[-1]
                val_indices = metrics.index[~np.isnan(metrics[f"validation/tre"])]
                columns = [c for c in metrics.columns if f"validation/" in c]
                metrics_v = metrics[columns]
                metrics_v.columns = [c.replace(f"validation/", "") for c in metrics_v.columns]
                self.summary = pd.concat([self.summary, metrics_v.loc[val_indices].iloc[-1]])
            
        
    def query(self, query: str):
        """Query the run using dictquery.
        
        Can be used to filter runs based on their configuration.
        """
        return dq.match(self.config, query)


class Dataset:
    """A dataset containing elastix runs.
    
    Args:
        project: Name of the project.
        runs: List of finished runs.
    """

    def __init__(self, project: str, runs: List[FinishedRun]) -> None:
        self.runs: List[FinishedRun] = runs
        self.project = project

    def add_run(self, run: FinishedRun):
        self.runs.append(run)

    def filter(self, query: str):
        """Filter the dataset using dictquery."""
        return Dataset(self.project, [run for run in self.runs if run.query(query)])

    def groupby(self, attrs: List[str]):
        """Group the runs by the given attributes."""
        if len(attrs) == 0:
            yield (), self.runs
        else:
            query_parts = [[] for _ in range(len(attrs))]
            unique_values = [[] for _ in range(len(attrs))]
            for i, attr in enumerate(attrs):
                for run in self.runs:
                    if attr in run.config:
                        value = run.config[attr]
                        if isinstance(value, list):
                            value = tuple(value)
                        unique_values[i].append(value)
                        if isinstance(value, str):
                            query_parts[i].append(f"{attr} == '{run.config[attr]}'")
                        else:
                            query_parts[i].append(f"{attr} == {run.config[attr]}")
                    else:
                        unique_values[i].append(None)
                        query_parts[i].append(f"NOT {attr}")

            for i in range(len(attrs)):
                query_parts[i] = list(dict.fromkeys(query_parts[i]))
                unique_values[i] = list(dict.fromkeys(unique_values[i]))

            for group, query_tuple in zip(itertools.product(*unique_values), itertools.product(*query_parts)):
                query = query_tuple[0]
                for i in range(1, len(query_tuple) - 1):
                    query += " AND " + query_tuple[i]
                query += " AND " + query_tuple[-1]
                yield group, self.filter(query).runs

    def aggregate(
        self,
        attrs: List[str] = [],
        metrics: List[str] = ["metric"],
        resolution: int = 0,
        val: bool = True,
    ):
        """Aggregate the runs by the given attributes.
        
        Args:
            attrs: List of attributes to group by.
            metrics: List of metrics to aggregate.
            resolution: Resolution to aggregate.
            val: Whether to aggregate the summaries or the iteration output. Defaults to summaries.
        """
        df = pd.DataFrame(columns=metrics)
        for group, runs in self.groupby(attrs):
            df_add = pd.DataFrame(columns=metrics)
            for run in runs:
                if val:
                    val_df = run.summary[metrics].to_frame()
                    val_df = val_df.transpose()
                    df_add = pd.concat([df_add, val_df])
                else:
                    df_add = pd.concat([df_add, run.resolutions[resolution][metrics]])
            for i, attr in enumerate(attrs):
                df_add[attr] = str(group[i])
            df = pd.concat([df, df_add])
        return df

    def aggregate_for_plot(self, attrs: List[str] = [], metric: str = "metric", resolution: int = 0):
        """Specialized aggregation function for convergence plots."""
        res = {}
        for group, runs in self.groupby(attrs):
            if len(runs) == 0:
                continue
            
            comb_arr = []
            max_size = len(runs[0].resolutions[resolution][metric])
            for run in runs:
                comb_arr.append(run.resolutions[resolution][metric].to_numpy())
                max_size = max(max_size, len(run.resolutions[resolution][metric]))
            for i in range(len(comb_arr)):
                comb_arr[i] = np.pad(comb_arr[i], (0,max_size - len(comb_arr[i])), "edge")
            comb_arr = np.array(comb_arr)
            avg_arr = np.mean(comb_arr, axis=0)
            median_arr = np.median(comb_arr, axis=0)
            std_arr = np.std(comb_arr, axis=0) * 2.0
            res[group] = (avg_arr, median_arr, std_arr)
        return res

    def save(self):
        """Save the dataset to disk."""
        path = DATASETS_PATH / f"{self.project}.pkl"
        with path.open("wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(project: str):
        """Load a dataset from disk."""
        path = DATASETS_PATH / f"{project}.pkl"
        if path.exists():
            with path.open("rb") as file:
                return pickle.load(file)
        return None
