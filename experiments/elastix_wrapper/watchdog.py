from pathlib import Path
import re
import threading
import time
import os
from typing import Any, Dict
import pandas as pd
import numpy as np
from comet_ml import Experiment


class SaveStrategy:
    def save(self, headers, values, resolution) -> None:
        pass

    def close(self) -> None:
        pass


class SaveStrategyFile(SaveStrategy):
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True)
        self.files = []

    def save(self, headers, values, resolution) -> None:
        if len(self.files) < resolution + 1:
            out_file = self.out_dir / f"{resolution}.dat"
            self.files.append(open(out_file.absolute().resolve(), "ab"))
        np.savetxt(self.files[resolution], values)

    def close(self) -> None:
        for file in self.files:
            file.close()


class SaveStrategyComet(SaveStrategy):
    def __init__(self, project: str, params: Dict[str, Any]) -> None:
        self.experiment: Experiment = Experiment(
            api_key=os.environ["COMET_API_KEY"], project_name=project
        )
        self.experiment.log_parameters(params)

    def save(self, headers, values, resolution) -> None:
        values[-1][-1] = np.sum(values[:, -1])
        to_log = dict(zip(headers[1:], values[-1][1:]))
        self.experiment.log_metrics(to_log, f"R{resolution}", values[-1][0])


class Watchdog(threading.Thread):
    def __init__(
        self,
        out_dir,
        n_resolutions,
        *args,
        **kwargs,
    ):
        super(Watchdog, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()
        self.set_input(out_dir, n_resolutions)

    def set_strategy(self, strategy: SaveStrategy):
        self.sv_strategy: SaveStrategy = strategy

    def set_input(self, out_dir, n_resolutions):
        self.out_dir = out_dir
        self.n_resolutions = n_resolutions

    def save_output(self, headers, values):
        pass

    def run(self):
        line_counts = [0 for _ in range(self.n_resolutions)]
        file_names = [
            self.out_dir / f"IterationInfo.0.R{r}.txt"
            for r in range(self.n_resolutions)
        ]
        r = 0

        while True:
            if not os.path.exists(file_names[r]):
                continue
            try:
                resolution_results = pd.read_csv(file_names[r], sep="	")
            except pd.errors.EmptyDataError:
                continue

            headers = resolution_results.columns.values
            headers = [re.sub(r"\d:", "", header).lower() for header in headers]
            values = resolution_results.values

            len_values = values.shape[0]
            len_diff = len_values - line_counts[r]
            line_counts[r] = len_values

            if len_diff > 0:
                self.sv_strategy.save(headers, values[len_values - len_diff :], r)
            elif r < self.n_resolutions - 1 and os.path.exists(file_names[r + 1]):
                r += 1
            elif self._stop_event.is_set():
                break

    def stop(self):
        self._stop_event.set()
