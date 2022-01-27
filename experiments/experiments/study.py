from operator import ge
import optuna
from optuna.samplers import NSGAIISampler

from experiments.wrapper import Wrapper
from experiments.parameters import Parameters, Collection

MAX_TIME_S = 60 * 8
STUDY_NAME = "pirpinia_gomea"
STUDY_DB = f"sqlite:///{STUDY_NAME}.db"


def get_params(trial):
    w0 = trial.suggest_float("Weight0", 0.0, 1.0)
    w1 = trial.suggest_float("Weight1", 0.0, 1000.0)
    params = (
        Parameters(mesh_size=8)
        .gomea(pop_size=80)
        .multi_metric(weight0=w0, weight1=w1)
        .instance(Collection.EMPIRE, 1)
        .stopping_criteria(max_time_s=MAX_TIME_S)
    )
    return params


def objective(trial):
    wrap = Wrapper()
    results = wrap.run(get_params(trial))
    return results["final_metric_0"], results["final_metric_1"]


if __name__ == "__main__":
    study = optuna.create_study(study_name=STUDY_NAME, storage=STUDY_DB, sampler=NSGAIISampler(
    ), load_if_exists=True, directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=None)
