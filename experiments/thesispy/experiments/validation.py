from typing import Collection
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.spatial import distance
from skimage.filters import threshold_multiotsu
import numpy as np
import SimpleITK as sitk
from thesispy.experiments.instance import RunResult
from thesispy.definitions import N_CORES, Collection
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import logging
import time

logger = logging.getLogger("Validation")

class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, desc=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        self._desc = desc
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            self._pbar.set_description(self._desc)
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def hessian(dvf_slice, p):
    p = np.array(p, dtype=int)
    try:
        dvf_slice[tuple(p)]
    except IndexError:
        print(f"Point {p} is out of bounds")
        return None

    n = len(p)
    output = np.matrix(np.zeros(n * n))
    output = output.reshape(n, n)
    max_indices = np.array(dvf_slice.shape) - 1
    ei = np.zeros(n, dtype=int)
    ej = np.zeros(n, dtype=int)

    for i in range(n):
        for j in range(i + 1):
            ei[i] = 1
            ej[j] = 1
            f1 = dvf_slice[tuple((np.clip(p + ei + ej, 0, max_indices)))]
            f2 = dvf_slice[tuple((np.clip(p + ei - ej, 0, max_indices)))]
            f3 = dvf_slice[tuple((np.clip(p - ei + ej, 0, max_indices)))]
            f4 = dvf_slice[tuple((np.clip(p - ei - ej, 0, max_indices)))]
            numdiff = (f1 - f2 - f3 + f4) / 4
            output[i, j] = numdiff
            output[j, i] = numdiff
            ei[i] = 0
            ej[j] = 0
    return output


def bending_energy_point(dvf, p):
    sum = 0.0
    for dim in range(len(dvf.shape) - 1):
        sum += np.square(np.linalg.norm(hessian(dvf[..., dim], p)))
    return sum


def bending_energy(dvf):
    results = ProgressParallel(n_jobs=N_CORES, desc="computing bending energy", backend="multiprocessing", total=np.prod(dvf.shape[:-1]))(
        delayed(bending_energy_point)(dvf, p) for p in np.ndindex(dvf.shape[:-1])
    )

    return np.sum(results) / np.prod(dvf.shape[:-1])


def dice_similarity(moving_deformed, fixed, levels):
    logger.info("Computing Dice Similarity")
    thresholds_moving = threshold_multiotsu(moving_deformed, classes=levels)
    thresholds_fixed = threshold_multiotsu(fixed, classes=levels)
    regions_moving_deformed = np.digitize(moving_deformed, bins=thresholds_moving)
    regions_fixed = np.digitize(fixed, bins=thresholds_fixed)

    intersection = np.sum((regions_moving_deformed == regions_fixed) & (regions_fixed > 0))
    union = np.sum(regions_moving_deformed > 0) + np.sum(regions_fixed > 0)
    return 2.0 * intersection / union


def hausdorff_distance(lms1, lms2, spacing=1):
    logger.info("Computing Hausdorff Distance")
    return np.max(distance.cdist(lms1, lms2).min(axis=1))


def dvf_rmse(dvf1, dvf2, spacing=1):
    logger.info("Computing DVF RMSE")
    return np.linalg.norm((dvf1 - dvf2) * spacing, axis=3).mean()


def mean_surface_distance(lms1, lms2, spacing=1):
    logger.info("Computing Mean Surface Distance")
    return np.mean(distance.cdist(lms1, lms2).min(axis=1) * spacing)


def tre(lms1, lms2, spacing=1):
    logger.info("Computing TRE")
    return np.linalg.norm((lms1 - lms2) * spacing, axis=1).mean()


def jacobian_determinant(dvf):
    logger.info("Computing Jacobian Determinant")
    axis_swap = 2 if len(dvf.shape) == 4 else 1
    dvf = np.swapaxes(dvf, 0, axis_swap)
    dvf_img = sitk.GetImageFromArray(dvf, isVector=True)
    jac_det_contracting_field = sitk.DisplacementFieldJacobianDeterminant(dvf_img)
    jac_det = sitk.GetArrayFromImage(jac_det_contracting_field)
    jac_det = np.swapaxes(jac_det, 0, axis_swap)
    if len(jac_det.shape) == 3:
        slice = jac_det.shape[-1] // 2
        jac_det = jac_det[..., slice]
    ax = sns.heatmap(jac_det, cmap="jet")
    return wandb.Image(ax.get_figure())


def get_cmap_color(cmap, f, a):
    c = cmap(f)
    c[..., 3] = a
    return c


def plot_voxels(
    data,
    lms=None,
    y_slice_depth=None,
    orientation=(0, -70),
    cmap_name="Greys",
    alpha=1.0,
):
    logger.info("Generating voxel plot")
    if len(data.shape) == 2:
        _, ax = plt.subplots(figsize =(7, 7))
        ax.imshow(data, cmap=cmap_name, alpha=alpha)
        return wandb.Image(ax.get_figure())
    
    if y_slice_depth is None:
        y_slice_depth = data.shape[1] // 2
    ax = plt.figure(figsize=(8, 8)).add_subplot(projection="3d")
    sliced_data = np.copy(data)
    sliced_data[:, :y_slice_depth, :] = 0

    cmap = cm.get_cmap(cmap_name)
    norm = Normalize(vmin=np.min(sliced_data), vmax=1.5 * np.max(sliced_data))

    colors = np.array(list(map(lambda x: get_cmap_color(cmap, norm(x), alpha), sliced_data)))

    ax.voxels(sliced_data, facecolors=colors, edgecolor=(0, 0, 0, 0.2))
    ax.set_xlim3d(1, 29)
    ax.set_ylim3d(5, 29)
    ax.set_zlim3d(1, 29)
    ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
    plt.locator_params(axis="y", nbins=3)
    ax.view_init(*orientation)

    if lms is not None:
        for lm in lms:
            if abs(lm[1] - y_slice_depth) <= 0.25:
                ax.scatter(lm[0], lm[1], lm[2], s=50, c="r")

    return wandb.Image(ax.get_figure())

def plot_dvf(data, scale=1.0, invert=False, slice=None):
    logger.info("Generating DVF plot")
    if len(data.shape[:-1]) > 2:
        if slice is None:
            slice = data.shape[0] // 2
        data = data[:,:,slice,:]
    
    X, Y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
    if invert:
        X = X + data[..., 1]
        Y = Y + data[..., 0]
        data = -data

    X = X + 0.5
    Y = Y + 0.5

    u = data[:,:,0]
    v = data[:,:,1]
    c = np.sqrt(u**2 + v**2)
    
    fig, ax = plt.subplots(figsize =(7, 7))
    qq = ax.quiver(X, Y, v, u, c, scale=scale, units='xy', angles='xy', scale_units='xy', cmap=plt.cm.jet)

    ax.set_xticks([i for i in range(data.shape[0])][::2])
    ax.set_yticks([i for i in range(data.shape[1])][::2])
    ax.set_xlim(0, data.shape[0])
    ax.set_ylim(0, data.shape[1])
    ax.grid(True)
    ax.set_aspect('equal')
    fig.colorbar(qq,fraction=0.045, pad=0.02, label='Displacement magnitude')
    return wandb.Image(ax.get_figure())

def plot_cpoints(points, grid_spacing, grid_origin, slice=None):
    logger.info("Generating control point plot")
    points_slice = points
    if len(points.shape) == 4:
        if slice is None:
            slice = points.shape[2] // 2
        points_slice = points[:, :, slice, :]

    grid_spacing = np.array(grid_spacing)
    grid_origin = np.array(grid_origin)

    grid_origin = grid_origin + 0.5
    X, Y = np.meshgrid(*[np.arange(grid_origin[i], grid_origin[i] + grid_spacing[i] * points_slice.shape[i], grid_spacing[i]) for i in range(len(points_slice.shape[:-1]))])
    colors = np.zeros(points_slice.shape[:-1])
    color = 0
    for p in np.ndindex(points_slice.shape[:-1]):
        colors[p] = color
        color += 1

    _, ax = plt.subplots(figsize =(7, 7))    
    cmap = plt.cm.coolwarm
    ax.scatter(X, Y, marker='+', c=colors, cmap=cmap, alpha=0.5, s=20)
    ax.scatter(points_slice[..., 0], points_slice[..., 1], marker='s', s=15, c=colors, cmap=cmap)
    return wandb.Image(ax.get_figure())


def calc_validation(result: RunResult):
    logger.info("Calculating validation metrics:")
    start = time.perf_counter()
    metrics = []
    levels = 2 if result.instance.collection == Collection.EXAMPLES else 3
    if result.dvf is not None:
        if result.instance.collection == Collection.SYNTHETIC:
            metrics.append({"validation/bending_energy": bending_energy(result.dvf)})
        metrics.append({"visualization/jacobian_determinant_slice": jacobian_determinant(result.dvf)})
        metrics.append({"visualization/dvf_slice": plot_dvf(result.dvf)})
        if result.instance.dvf is not None:
            metrics.append({"validation/dvf_rmse": dvf_rmse(result.dvf, result.instance.dvf)})
    if result.deformed is not None:
        metrics.append({"validation/dice_similarity": dice_similarity(result.deformed, result.instance.fixed, levels)})
        if result.instance.collection == Collection.SYNTHETIC:
            metrics.append({"visualization/deformed_image_slice": plot_voxels(result.deformed)})
    if result.deformed_lms is not None:
        metrics.append({"validation/hausdorff_distance": hausdorff_distance(result.deformed_lms, result.instance.lms_moving)})
        metrics.append({"validation/mean_surface_distance": mean_surface_distance(result.deformed_lms, result.instance.lms_moving)})
        metrics.append({"validation/tre": tre(result.deformed_lms, result.instance.lms_moving)})
    if result.control_points is not None:
        metrics.append({"visualization/cpoints_slice": plot_cpoints(result.control_points, result.grid_spacing, result.grid_origin)})

    end = time.perf_counter()
    logger.info(f"Validation metrics calculated in {end - start:.2f}s")

    return metrics
