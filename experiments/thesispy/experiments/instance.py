from thesispy.definitions import *
import SimpleITK as sitk
import numpy as np
from dataclasses import dataclass


@dataclass
class Instance:
    collection: Collection
    instance: int
    moving: np.ndarray
    moving_path: Path
    fixed: np.ndarray
    spacing: np.ndarray
    origin: np.ndarray
    lms_moving: np.ndarray = None
    lms_fixed: np.ndarray = None
    lms_fixed_path: Path = None
    dvf: np.ndarray = None

@dataclass
class RunResult:
    instance: Instance
    deformed: np.ndarray = None
    deformed_lms: np.ndarray = None
    dvf: np.ndarray = None
    control_points: np.ndarray = None


def get_np_array(img_path: Path):
    img = sitk.ReadImage(str(img_path.resolve()))
    data = sitk.GetArrayFromImage(img)
    if len(data.shape) == 4 or (len(data.shape) == 3 and data.shape[-1] > 2):
        data = np.swapaxes(data, 0, 2)
    else:
        data = np.swapaxes(data, 0, 1)
    return data.astype(np.float64)


def load_imgs(collection: Collection, instance: int):
    config = INSTANCES_CONFIG[collection.value]
    path_moving = INSTANCES_SRC / config["folder"] / "scans" / f"{instance:02}_Moving.{config['extension']}"
    path_fixed = INSTANCES_SRC / config["folder"] / "scans" / f"{instance:02}_Fixed.{config['extension']}"
    moving = get_np_array(path_moving)
    fixed = get_np_array(path_fixed)
    spacing = sitk.ReadImage(str(path_moving.resolve())).GetSpacing()
    origin = sitk.ReadImage(str(path_moving.resolve())).GetOrigin()
    return moving, path_moving, fixed, spacing, origin


def get_instance(collection: Collection, instance_id: int):
    instance = Instance(collection, instance_id, *load_imgs(collection, instance_id))
    config = INSTANCES_CONFIG[collection.value]
    if config["landmarks"]:
        path_lms_moving = INSTANCES_SRC / config["folder"] / "landmarks" / f"{instance_id:02}_Moving.txt"
        path_lms_fixed = INSTANCES_SRC / config["folder"] / "landmarks" / f"{instance_id:02}_Fixed.txt"
        instance.lms_moving = np.loadtxt(path_lms_moving, skiprows=2)
        instance.lms_fixed = np.loadtxt(path_lms_fixed, skiprows=2)
        instance.lms_fixed_path = path_lms_fixed
    if config["dvf"] and config["dvf_indices"][instance_id - 1]:
        path_dvf = INSTANCES_SRC / config["folder"] / "dvf" / f"{instance_id:02}.npy"
        instance.dvf = np.load(path_dvf)
    return instance

def read_deformed_lms(path: Path):
    deformed_lms = []
    with open(path) as file:
        lines = file.readlines()
        for _, line in enumerate(lines):
            s = line.split(";")[3]
            s = s[s.find("[") + 1 : s.find("]")].split(" ")
            index = np.array([float(s[1]), float(s[2]), float(s[3])])
            deformed_lms.append(index)
    return np.array(deformed_lms)

def read_controlpoints(path: Path):
    with open(path) as file:
        lines = file.readlines()
        dim = len(lines[0].split()) // 2
        grid_size = lines[-1].split()[:dim]
        grid = np.zeros((int(grid_size[0]) + 1, int(grid_size[1]) + 1, int(grid_size[2]) + 1, dim))
        for line in lines:
            s = line.split()
            index = np.array(s[:dim], dtype=int)
            point = np.array(s[dim:], dtype=float)
            grid[index[0], index[1], index[2]] = point
        return grid
