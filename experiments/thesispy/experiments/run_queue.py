import sys
import logging
from pathlib import Path

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from thesispy.experiments.experiment import ExperimentQueue, run_experiment

logger = logging.getLogger("Queue")
exp_queue = ExperimentQueue()

while exp_queue.peek():
    experiment = exp_queue.pop()
    if not run_experiment(experiment):
        logger.warning(f"Experiment from {experiment.project} failed.")
        logger.warning(f"Experiment details: {experiment}")