from pathlib import Path
import shutil
import threading
import time
import wandb
import os
import pandas as pd


class Watchdog(threading.Thread):
    def __init__(self, out_dir, n_resolutions,  *args, **kwargs):
        super(Watchdog, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()
        self.set_output(out_dir, n_resolutions)

    def set_output(self, out_dir, n_resolutions):
        self.out_dir = out_dir
        self.n_resolutions = n_resolutions

    def run(self):
        line_counts = [0 for _ in range(self.n_resolutions)]
        file_names = [
            self.out_dir / f"IterationInfo.0.R{r}.txt" for r in range(self.n_resolutions)]
        while not self._stop_event.is_set():
            time.sleep(0.1)
            for r in range(self.n_resolutions):
                if not os.path.exists(file_names[r]):
                    continue
                
                try:
                    resolution_results = pd.read_csv(
                        file_names[r], sep="	").to_dict()
                except pd.errors.EmptyDataError:
                    continue

                headers = list(resolution_results.keys())
                values = [list(v.values())
                          for v in resolution_results.values()]

                len_values = len(values[0])
                len_diff = len_values - line_counts[r]
                line_counts[r] = len_values
                for line in range(len_values-len_diff, len_values):
                    for index, header in enumerate(headers):
                        it_nr = values[0][line]
                        # wandb.log({header: values[index][-1 * line]}, step=it_nr)
                        print(f"{it_nr}: {header}={values[index][line]}")

    def stop(self):
        time.sleep(2.0)
        self._stop_event.set()
