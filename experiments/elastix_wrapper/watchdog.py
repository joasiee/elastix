import re
import threading
import time
from more_itertools import first
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

    @staticmethod
    def read_file(filename):
        with open(filename, "rb") as file:
            first_line, last_line = None, None
            try:
                first_line = file.readline().decode().split()
                file.seek(-2, os.SEEK_END)
                while file.read(1) != b'\n':
                    file.seek(-2, os.SEEK_CUR)
                last_line = file.readline().decode().split()
            except OSError:
                pass
            return first_line, last_line

    def run(self):
        line_counts = [0 for _ in range(self.n_resolutions)]
        file_names = [
            self.out_dir / f"IterationInfo.0.R{r}.txt" for r in range(self.n_resolutions)]
        r = 0

        while not self._stop_event.is_set():
            time.sleep(0.1)
            
            if not os.path.exists(file_names[r]):
                continue

            headers, last_line = Watchdog.read_file(file_names[r])
            if headers and last_line and len(headers) == len(last_line):
                uptodate = line_counts[r] == int(last_line[0])

                if r < self.n_resolutions - 1 and os.path.exists(file_names[r+1]) and uptodate:
                    r += 1

                if not uptodate:
                    line_counts[r] = int(last_line[0])
                    scalars = {}
                    for index, header in enumerate(headers[1:]):
                        header = re.sub(r'\d:', '', header).lower()
                        scalars[f"R{r}/"+header] = float(last_line[index+1])
                    wandb.log(scalars)

    def stop(self):
        time.sleep(1.0)
        self._stop_event.set()
