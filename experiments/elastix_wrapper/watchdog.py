import re
import threading
import time
import os
import pandas as pd

class Watchdog(threading.Thread):
    def __init__(self, out_dir, n_resolutions, *args, **kwargs):
        super(Watchdog, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()
        self.set_output_dir(out_dir, n_resolutions)

    def set_output_dir(self, out_dir, n_resolutions):
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
                resolution_results = pd.read_csv(file_names[r], sep="	")
            except pd.errors.EmptyDataError:
                continue

            headers = resolution_results.columns.values[1:]
            headers = [re.sub(r"\d:", "", header).lower() for header in headers]
            values = resolution_results.values[:, 1:]

            len_values = values.shape[0]
            len_diff = len_values - line_counts[r]
            line_counts[r] = len_values

            if len_diff > 0:
                yield headers, values[len_values - len_diff :]
            elif r < self.n_resolutions - 1 and os.path.exists(file_names[r + 1]):
                r += 1
            elif self._stop_event.is_set():
                break

            time.sleep(0.1)

    def stop(self):
        time.sleep(5)
        self._stop_event.set()
