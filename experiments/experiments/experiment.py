import json
from typing import List
import redis
from dotenv import load_dotenv
import os
import wandb
from sshtunnel import SSHTunnelForwarder

from elastix_wrapper.parameters import Parameters

WANDB_ENTITY = "joasiee"


class Experiment:
    __slots__ = ["project", "params"]

    def __init__(self, project: str, params: Parameters) -> None:
        params.prune()
        self.project = project
        self.params = params

    @classmethod
    def from_json(cls, jsondump):
        pyjson = json.loads(jsondump)
        params = Parameters(pyjson["params"]).set_paths()
        return cls(pyjson["project"], params)

    def already_done(self, keys: List[str]):
        keys += ["Optimizer", "Instance"]
        filters = {"state": {"$in": ["finished", "running"]}}
        for key in keys:
            filters[f"config.{key}"] = self.params[key]
        api = wandb.Api()
        runs = api.runs(WANDB_ENTITY + "/" + self.project, filters=filters)
        return len(runs) > 0

    def to_json(self):
        return json.dumps({
            "project": self.project,
            "params": self.params.params
        })

    def __str__(self) -> str:
        return self.to_json()


class ExperimentQueue:
    queue_id = 'queue:experiments'

    def __init__(self) -> None:
        load_dotenv()
        self.ssh_forwarding_enable()
        self.client = redis.Redis(host="localhost", port=self.local_port, db=0)

    def ssh_forwarding_enable(self):
        self.sshserver = SSHTunnelForwarder(
            os.environ["REDIS_HOST"],
            ssh_username="root",
            remote_bind_address=('127.0.0.1', 6379)
        )
        self.sshserver.start()
        self.local_port = self.sshserver.local_bind_port

    def push(self, experiment: Experiment) -> None:
        self.client.rpush(ExperimentQueue.queue_id, experiment.to_json())

    def pop(self) -> Experiment:
        packed = self.client.lpop(ExperimentQueue.queue_id)
        if packed:
            return Experiment.from_json(packed)
        return None

    def peek(self) -> Experiment:
        return self.client.lrange(ExperimentQueue.queue_id, 0, 0)

    def size(self) -> int:
        return self.client.llen(ExperimentQueue.queue_id)

    def clear(self) -> None:
        self.client.delete(ExperimentQueue.queue_id)


if __name__ == "__main__":
    expq = ExperimentQueue()
    # expq.clear()
    print(expq.size())
