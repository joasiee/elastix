import sys
from pathlib import Path

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from thesispy.experiments.experiment import ExperimentQueue, run_experiment

exp_queue = ExperimentQueue()

while exp_queue.peek():
    experiment = exp_queue.pop()
    run_experiment(experiment)