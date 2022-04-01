import re
import threading
import time
import wandb
import os
import pandas as pd


class Watchdog(threading.Thread):
    def __init__(self, out_dir, n_resolutions, *args, **kwargs):
        super(Watchdog, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()
        self.set_output(out_dir, n_resolutions)

    def set_output(self, out_dir, n_resolutions):
        self.out_dir = out_dir
        self.n_resolutions = n_resolutions

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
                resolution_results = pd.read_csv(file_names[r], sep="	").to_dict()
            except pd.errors.EmptyDataError:
                continue

            headers = list(resolution_results.keys())[1:]
            values = [list(v.values()) for v in resolution_results.values()][1:]

            len_values = len(values[0])
            len_diff = len_values - line_counts[r]
            line_counts[r] = len_values

            for line in range(len_values - len_diff, len_values):
                scalars = {}
                for index, header in enumerate(headers):
                    header = re.sub(r"\d:", "", header).lower()
                    scalars[f"R{r}/" + header] = values[index][line]
                wandb.log(scalars)

            if self._stop_event.is_set() and r == self.n_resolutions - 1:
                break

            if (
                r < self.n_resolutions - 1
                and os.path.exists(file_names[r + 1])
                and len_diff == 0
            ):
                r += 1

            time.sleep(5)

    def stop(self):
        time.sleep(1)
        self._stop_event.set()
