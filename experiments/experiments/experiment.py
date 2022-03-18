import json
from typing import Any, Dict
import redis
from dotenv import load_dotenv
import os


class Experiment:
    __slots__ = ["project", "params"]

    def __init__(self, project: str, params: Dict[str, Any]) -> None:
        self.project = project
        self.params = params

    @classmethod
    def from_json(cls, jsondump):
        pyjson = json.loads(jsondump)
        return cls(pyjson["project"], pyjson["params"])

    def to_json(self):
        return json.dumps({
            "project": self.project,
            "params": self.params
        })

    def __str__(self) -> str:
        return self.to_json()


class ExperimentQueue:
    queue_id = 'queue:experiments'

    def __init__(self) -> None:
        load_dotenv()
        self.client = redis.Redis(
            host=os.environ["REDIS_HOST"], port=6379, password=os.environ["REDIS_PWD"], username="joasiee", db=0)

    def push(self, experiment: Experiment) -> None:
        self.client.rpush(ExperimentQueue.queue_id, experiment.to_json())

    def pop(self) -> Experiment:
        packed = self.client.lpop(ExperimentQueue.queue_id)
        if packed:
            return Experiment.from_json(packed)
        return None


if __name__ == "__main__":
    expq = ExperimentQueue()
    expq.push(Experiment("test", {"param1": 0.01}))
    x1= expq.pop()
    x2 = expq.pop()
    print("")
