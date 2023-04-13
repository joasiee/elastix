from typing import List
from thesispy.definitions import *
import SimpleITK as sitk
import numpy as np
from dataclasses import dataclass


@dataclass
class Instance:
    """An instance of a collection.
    
    All relevant data is stored in an instantation of this class.
    """
    collection: Collection
    instance: int
    moving: np.ndarray
    moving_path: Path
    fixed: np.ndarray
    spacing: np.ndarray
    origin: np.ndarray
    direction: np.ndarray
    size: np.ndarray
    lms_moving: np.ndarray = None
    lms_fixed: np.ndarray = None
    lms_fixed_path: Path = None
    surface_points: List[np.ndarray] = None
    surface_points_paths: List[Path] = None
    dvf: np.ndarray = None
    mask: np.ndarray = None
    moving_mask_path: Path = None


@dataclass
class RunResult:
    """The result of a registration run."""
    instance: Instance
    deformed: np.ndarray = None
    deformed_mask: np.ndarray = None
    deformed_lms: np.ndarray = None
    deformed_surface_points: List[np.ndarray] = None
    dvf: np.ndarray = None
    control_points: np.ndarray = None
    grid_spacing: np.ndarray = None
    grid_origin: np.ndarray = None
    bending_energy: float = None
    transform_params: Path = None

@dataclass
class MORunResult:
    """The result of a registration run with MO-RV-GOMEA."""
    instance: Instance
    approximation_set: List[RunResult] = None
    objective_values: np.ndarray = None
    constraint_values: np.ndarray = None
    number_of_objectives: int = None
    

def get_np_array(img_path: Path):
    img = sitk.ReadImage(str(img_path.resolve()))
    data = sitk.GetArrayFromImage(img)
    return data.astype(np.float64)


def load_imgs(collection: Collection, instance: int):
    config = INSTANCES_CONFIG[collection.value]
    path_moving = INSTANCES_SRC / config["folder"] / "scans" / f"{instance:02}_Moving.{config['extension']}"
    path_fixed = INSTANCES_SRC / config["folder"] / "scans" / f"{instance:02}_Fixed.{config['extension']}"
    moving = get_np_array(path_moving)
    fixed = get_np_array(path_fixed)
    img = sitk.ReadImage(str(path_moving.resolve()))
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = np.array(img.GetDirection()).reshape((len(spacing), len(spacing)))
    size = np.array(img.GetSize())
    return moving, path_moving, fixed, spacing, origin, direction, size


def get_instance(collection: Collection, instance_id: int):
    instance = Instance(collection, instance_id, *load_imgs(collection, instance_id))
    config = INSTANCES_CONFIG[collection.value]
    if config["landmarks"]:
        path_lms_moving = INSTANCES_SRC / config["folder"] / "landmarks" / f"{instance_id:02}_Moving.txt"
        path_lms_fixed = INSTANCES_SRC / config["folder"] / "landmarks" / f"{instance_id:02}_Fixed.txt"
        instance.lms_moving = np.loadtxt(path_lms_moving, skiprows=2)
        instance.lms_fixed = np.loadtxt(path_lms_fixed, skiprows=2)
        instance.lms_fixed_path = path_lms_fixed
    if config["surface_points"]:
        instance.surface_points, instance.surface_points_paths = [], []
        path_surface_points = INSTANCES_SRC / config["folder"] / "landmarks" / "surfaces"
        for file in sorted(path_surface_points.iterdir()):
            if file.name.startswith(f"{instance_id:02}_Moving"):
                instance.surface_points.append(np.loadtxt(file, skiprows=2))
            elif file.name.startswith(f"{instance_id:02}_Fixed"):
                instance.surface_points_paths.append(file)
    if config["dvf"] and config["dvf_indices"][instance_id - 1]:
        path_dvf = INSTANCES_SRC / config["folder"] / "dvf" / f"{instance_id:02}.npy"
        instance.dvf = np.load(path_dvf)
    if config["masks"]:
        path_mask = INSTANCES_SRC / config["folder"] / "masks" / f"{instance_id:02}_Fixed.{config['extension']}"
        instance.moving_mask_path = (
            INSTANCES_SRC / config["folder"] / "masks" / f"{instance_id:02}_Moving.{config['extension']}"
        )
        instance.mask = get_np_array(path_mask)
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
    if not path.exists():
        return None
    with open(path) as file:
        lines = file.readlines()
        dim = len(lines[0].split()) // 2
        grid_size = lines[-1].split()[:dim]
        grid = np.zeros(tuple([int(grid_size[i]) + 1 for i in range(dim)] + [dim]))
        for line in lines:
            s = line.split()
            index = np.array(s[:dim], dtype=int)
            point = np.array(s[dim:], dtype=float)
            grid[tuple(index)] = point
        return grid


def read_transform_params(transform_params_file: Path):
    with open(transform_params_file) as file:
        dim, grid_spacing, grid_origin = None, None, None
        for line in file.readlines():
            split = line.split()
            if len(split) >= 2:
                if split[0][1:] == "FixedImageDimension":
                    dim = int(split[1][:-1])
                if split[0][1:] == "GridOrigin":
                    grid_origin = split[1:]
                    grid_origin[-1] = grid_origin[-1][:-1]
                    grid_origin = np.array(grid_origin, dtype=float)
                if split[0][1:] == "GridSpacing":
                    grid_spacing = split[1:]
                    grid_spacing[-1] = grid_spacing[-1][:-1]
                    grid_spacing = np.array(grid_spacing, dtype=float)

        return dim, grid_spacing, grid_origin
