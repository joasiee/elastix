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
from matplotlib.colors import Normalize, LinearSegmentedColormap
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
    results = ProgressParallel(
        n_jobs=N_CORES, desc="computing bending energy", backend="multiprocessing", total=np.prod(dvf.shape[:-1])
    )(delayed(bending_energy_point)(dvf, p) for p in np.ndindex(dvf.shape[:-1]))
    be = np.sum(results) / np.prod(dvf.shape[:-1])
    logger.info(f"Bending Energy: {be}")
    return be


def set_similarity(moving_deformed, fixed, levels, type="dice"):
    thresholds_moving = threshold_multiotsu(moving_deformed, classes=levels)
    thresholds_fixed = threshold_multiotsu(fixed, classes=levels)
    regions_moving_deformed = np.digitize(moving_deformed, bins=thresholds_moving)
    regions_fixed = np.digitize(fixed, bins=thresholds_fixed)

    similarities = []
    for region_value in np.unique(regions_fixed):
        intersection = np.sum((regions_moving_deformed == regions_fixed) & (regions_fixed == region_value))
        sum_pixels = np.sum(regions_fixed == region_value) * 2
        if type == "dice":
            similarities.append(2.0 * intersection / sum_pixels)
        elif type == "jaccard":
            similarities.append(intersection / (sum_pixels - intersection))
    
    similarity = np.mean(similarities)
    logger.info(f"{type.capitalize()} Similarity: {similarity}")
    return similarity

def hausdorff_distance(surface_points, surface_points_deformed, spacing=1):
    distances = []
    for i in range(len(surface_points)):
        distances.append(np.max(distance.cdist(surface_points[i], surface_points_deformed[i]).min(axis=1)) * spacing)
    hdist = np.mean(distances)
    logger.info(f"Weighted Hausdorff Distance: {hdist}")
    return hdist


def dvf_rmse(dvf1, dvf2, spacing=1):
    rmse = np.linalg.norm((dvf1 - dvf2) * spacing, axis=3).mean()
    logger.info(f"DVF RMSE: {rmse}")
    return rmse


def mean_surface_distance(surface_points, surface_points_deformed, spacing=1):
    distances = []
    for i in range(len(surface_points)):
        distances.append(np.mean(distance.cdist(surface_points[i], surface_points_deformed[i]).min(axis=1) * spacing))
    msd = np.mean(distances)
    logger.info(f"Weighted Mean Surface Distance: {msd}")
    return msd


def tre(lms1, lms2, spacing=1):
    tre = np.linalg.norm((lms1 - lms2) * spacing, axis=1).mean()
    logger.info(f"TRE: {tre}")
    return tre


def jacobian_determinant(dvf, fig=None):
    axis_swap = 2 if len(dvf.shape) == 4 else 1
    dvf = np.swapaxes(dvf, 0, axis_swap)
    dvf_img = sitk.GetImageFromArray(dvf, isVector=True)
    jac_det_contracting_field = sitk.DisplacementFieldJacobianDeterminant(dvf_img)
    jac_det = sitk.GetArrayFromImage(jac_det_contracting_field)
    jac_det = np.swapaxes(jac_det, 0, axis_swap)
    if len(jac_det.shape) == 3:
        slice = jac_det.shape[-1] // 2
        jac_det = jac_det[..., slice]

    if fig is not None:
        ax = fig.add_subplot(2, 2, 4)
        sns.heatmap(jac_det, cmap="jet", ax=ax, square=True, cbar_kws={"fraction": 0.045, "pad": 0.02})
    else:
        sns.heatmap(jac_det, cmap="jet", square=True)
    ax.invert_yaxis()
    logger.info(f"Jacobian min,max: {np.min(jac_det)}, {np.max(jac_det)}")


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
    fig=None,
):
    if len(data.shape) == 2:
        if fig is None:
            _, ax = plt.subplots(figsize=(7, 7))
        else:
            ax = fig.add_subplot(2, 2, 1)
        ax.imshow(data, cmap=cmap_name, alpha=alpha)

    if y_slice_depth is None:
        y_slice_depth = data.shape[1] // 2
    
    if fig is None:
        ax = plt.figure(figsize=(8, 8)).add_subplot(projection="3d")
    else:
        ax = fig.add_subplot(2, 2, 1, projection="3d")

    sliced_data = np.copy(data)
    sliced_data[:, :y_slice_depth, :] = 0
    sliced_data[sliced_data < 5] = 0

    cmap = cm.get_cmap(cmap_name)
    norm = Normalize(vmin=np.min(sliced_data), vmax=1.5 * np.max(sliced_data))

    colors = np.array(list(map(lambda x: get_cmap_color(cmap, norm(x), alpha), sliced_data)))
    
    ax.voxels(sliced_data, facecolors=colors, edgecolor=(0, 0, 0, 0.2))
    ax.set_xlim3d(1, 29)
    ax.set_ylim3d(5, 29)
    ax.set_zlim3d(1, 29)
    ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
    ax.locator_params(axis="y", nbins=3)
    ax.view_init(*orientation)

    if lms is not None:
        for lm in lms:
            if abs(lm[1] - y_slice_depth) <= 0.25:
                ax.scatter(lm[0], lm[1], lm[2], s=50, c="r")


def plot_dvf(data, scale=1, invert=False, slice=None, fig=None):
    if len(data.shape[:-1]) > 2:
        if slice is None:
            slice = data.shape[0] // 2
        data = data[:, :, slice, :]

    X, Y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
    if invert:
        X = X + data[..., 1]
        Y = Y + data[..., 0]
        data = -data

    X = X + 0.5
    Y = Y + 0.5

    u = data[:, :, 0]
    v = data[:, :, 1]
    c = np.sqrt(u**2 + v**2)

    if fig is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        ax = fig.add_subplot(2, 2, 3)

    qq = ax.quiver(X, Y, v, u, c, scale=scale, units="xy", angles="xy", scale_units="xy", cmap=plt.cm.jet)

    ax.set_xlim(0, data.shape[0])
    ax.set_ylim(0, data.shape[1])
    ax.set_aspect("equal")
    fig.colorbar(qq, fraction=0.045, pad=0.02, label="Displacement magnitude", ax=ax)


def plot_cpoints(points, grid_spacing, grid_origin, slice=None, fig=None):
    points_slice = points
    if len(points.shape) == 4:
        if slice is None:
            slice = points.shape[2] // 2
        points_slice = points[:, :, slice, :]

    grid_spacing = np.array(grid_spacing)
    grid_origin = np.array(grid_origin)

    grid_origin = grid_origin + 0.5
    X, Y = np.meshgrid(
        *[
            np.arange(grid_origin[i], grid_origin[i] + grid_spacing[i] * points_slice.shape[i], grid_spacing[i])
            for i in range(len(points_slice.shape[:-1]))
        ]
    )

    colors = np.zeros(points_slice.shape[:-1])
    x_slice = colors.shape[0] // 2
    y_slice = colors.shape[1] // 2
    colors[:x_slice, :y_slice] = 0.25
    colors[x_slice:, :y_slice] = 0.5
    colors[:x_slice, y_slice:] = 0.75
    colors[x_slice:, y_slice:] = 1.0
            
    colormap_colors = ['#ffcc00', 'red', 'green', 'blue']
    cmap = LinearSegmentedColormap.from_list('quadrants', colormap_colors)

    if fig is None:
        _, ax = plt.subplots(figsize=(7, 7))
    else:
        ax = fig.add_subplot(2, 2, 2)

    ax.scatter(Y, X, marker="+", c=colors, cmap=cmap, alpha=0.3, s=20)
    ax.scatter(points_slice[..., 0], points_slice[..., 1], marker="s", s=15, c=colors, cmap=cmap, alpha=0.8)

def calc_validation(result: RunResult):
    logger.info("Calculating validation metrics:")
    start = time.perf_counter()
    metrics = []
    fig = plt.figure(figsize=(8, 8))
    levels = 2 if result.instance.collection == Collection.EXAMPLES else 3
    if result.dvf is not None:
        if result.instance.collection == Collection.SYNTHETIC:
            metrics.append({"validation/bending_energy": bending_energy(result.dvf)})
        jacobian_determinant(result.dvf, fig=fig)
        plot_dvf(result.dvf, fig=fig)
        if result.instance.dvf is not None:
            metrics.append({"validation/dvf_rmse": dvf_rmse(result.dvf, result.instance.dvf)})
    if result.deformed is not None:
        metrics.append({"validation/dice_similarity": set_similarity(result.deformed, result.instance.fixed, levels)})
        metrics.append({"validation/jaccard_similarity": set_similarity(result.deformed, result.instance.fixed, levels, "jaccard")})
        if result.instance.collection == Collection.SYNTHETIC:
            plot_voxels(result.deformed, fig=fig)
    if result.deformed_lms is not None:
        metrics.append({"validation/hausdorff_distance": hausdorff_distance(result.instance.surface_points, result.deformed_surface_points)})
        metrics.append({"validation/mean_surface_distance": mean_surface_distance(result.instance.surface_points, result.deformed_surface_points)})
        metrics.append({"validation/tre": tre(result.deformed_lms, result.instance.lms_moving)})
    if result.control_points is not None:
        plot_cpoints(result.control_points, result.grid_spacing, result.grid_origin, fig=fig)

    plt.tight_layout()
    metrics.append({"visualization/slices": wandb.Image(fig)})

    end = time.perf_counter()
    logger.info(f"Validation metrics calculated in {end - start:.2f}s")

    return metrics
