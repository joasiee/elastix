from enum import Enum
import json
import os
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).parent.parent
IMG_DIR = ROOT_DIR / "notebooks" / "img"
BASE_PARAMS_PATH = ROOT_DIR / Path("resources", "base_params.json")
N_CORES = int(os.environ["OMP_NUM_THREADS"])
INSTANCE_CONFIG_PATH = ROOT_DIR / Path("resources", "instances.json")
INSTANCES_CONFIG: Dict[str, str] = {}
INSTANCES_SRC = Path(os.environ.get("INSTANCES_SRC"))
INSTANCES_CONFIG: Dict[str, str] = {}

with INSTANCE_CONFIG_PATH.open() as f:
    INSTANCES_CONFIG = json.loads(f.read())

class Collection(str, Enum):
    EMPIRE = "EMPIRE"
    LEARN = "LEARN"
    EXAMPLES = "EXAMPLES"
    SYNTHETIC = "SYNTHETIC"

class GOMEAType(Enum):
    GOMEA_FULL = -1
    GOMEA_UNIVARIATE = 1
    GOMEA_CP = -6