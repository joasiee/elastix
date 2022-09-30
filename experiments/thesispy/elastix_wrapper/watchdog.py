from pathlib import Path
import re
import shutil
import threading
import time
import wandb
import os
import pandas as pd
import numpy as np


class SaveStrategy:
    def save(self, headers, row, resolution) -> None:
        pass

    def save_custom(self, obj) -> None:
        pass

    def close(self) -> None:
        pass


class SaveStrategyPrint(SaveStrategy):
    def save(self, headers, row, resolution) -> None:
        print(f"R{resolution}: {headers} --- {row}")


class SaveStrategyWandb(SaveStrategy):
    def __init__(self, experiment, run_dir: Path, batch_size: int = 1) -> None:
        wandb.init(project=experiment.project, name=str(experiment.params), reinit=True)
        wandb.config.update(experiment.params.params)
        self.run_dir = run_dir
        self.batch_size = batch_size
        self._rowcount = -1
        self._sum_time = 0
        self._resolution = 0
        self._buffer = (None, None)

    def _reset_state(self):
        self._rowcount = 0
        self._sum_time = 0
        self._resolution = 0
        self._buffer = (None, None)

    def _log_buffer(self):
        if self._buffer[0] is not None:
            headers, row = self._buffer
            row[-1] = self._sum_time
            headers = [f"R{self._resolution}/{header}" for header in headers]
            metrics = dict(zip(headers, row))
            wandb.log(metrics)
            self._reset_state()

    def save(self, headers, row, resolution) -> None:
        self._rowcount += 1
        self._sum_time += row[-1]
        self._buffer = (headers, row)
        self._resolution = resolution

        if self._rowcount % self.batch_size == 0:
            self._log_buffer()

    def save_custom(self, obj) -> None:
        wandb.log(obj, commit=False)

    def close(self) -> None:
        self._log_buffer()
        wandb.save(str((self.run_dir / "out" / "TransformParameters.0.txt").resolve()))
        wandb.save(str((self.run_dir / "out" / "controlpoints.dat").resolve()))
        wandb_dir = Path(wandb.run.dir)
        wandb.finish()
        shutil.rmtree(self.run_dir.absolute())
        shutil.rmtree(wandb_dir.parent.absolute())


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

    def run(self):
        line_counts = [0 for _ in range(self.n_resolutions)]
        file_names = [self.out_dir / f"IterationInfo.0.R{r}.txt" for r in range(self.n_resolutions)]
        r = 0

        while True:
            time.sleep(1)

            if not os.path.exists(file_names[r]):
                continue
            try:
                resolution_results = pd.read_csv(file_names[r], sep="	")
            except pd.errors.EmptyDataError:
                continue

            if np.count_nonzero(resolution_results.isnull().values) > 0:
                continue

            headers = resolution_results.columns.values
            headers = [re.sub(r"\d:", "", header).lower() for header in headers]
            values = resolution_results.values

            len_values = values.shape[0]
            len_diff = len_values - line_counts[r]
            line_counts[r] = len_values

            if len_diff > 0:
                for row in values[len_values - len_diff :]:
                    self.sv_strategy.save(headers, row, r)
            elif r < self.n_resolutions - 1 and os.path.exists(file_names[r + 1]):
                r += 1
            elif self._stop_event.is_set():
                break

    def stop(self):
        self._stop_event.set()
