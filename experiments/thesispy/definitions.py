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
    """Registration collection type."""
    EMPIRE = "EMPIRE"
    LEARN = "LEARN"
    EXAMPLES = "EXAMPLES"
    SYNTHETIC = "SYNTHETIC"


class LinkageType(Enum):
    """GOMEA linkage type."""
    FULL = -1
    UNIVARIATE = 1
    CP_MARGINAL = -6
    STATIC_EUCLIDEAN = -3

class RedistributionMethod(Enum):
    """GOMEA-LS redistribution method."""
    Random = 0
    BestN = 1

class IterationSchedule(Enum):
    """GOMEA-LS iteration schedule."""
    Static = 0
    Logarithmic = 1


STATIC_LINKAGE_MAPPING = {LinkageType.STATIC_EUCLIDEAN: 1}
