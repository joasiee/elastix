from typing import Collection
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.spatial import distance
from scipy.ndimage import zoom
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
VALIDATION_NAMES = [
    "tre",
    "mean_surface_distance",
    "hausdorff_distance",
    "dice_similarity",
    "bending_energy",
    "dvf_rmse",
]
VALIDATION_NAMES_NEW = [
    "tre",
    "mean_surface_cube",
    "mean_surface_sphere",
    "hausdorff_cube",
    "hausdorff_sphere",
    "bending_energy",
    "dvf_rmse",
]
VALIDATION_ABBRVS = ["$TRE$", "$WASD$", "$WHD$", "$DSC$", "$E_b$", r"$\vec{v}_{\epsilon}$"]
VALIDATION_ABBRVS_NEW = [
    "$TRE$",
    "$ASD_{\\textsc{cube}}$",
    "$ASD_{\\textsc{sphere}}$",
    "$HD_{\\textsc{cube}}$",
    "$HD_{\\textsc{sphere}}$",
    "$E_b$",
    "$\\vec{v}_{\\epsilon}$",
]


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
        n_jobs=N_CORES,
        desc="computing bending energy",
        backend="multiprocessing",
        total=np.prod(dvf.shape[:-1]),
    )(delayed(bending_energy_point)(dvf, p) for p in np.ndindex(dvf.shape[:-1]))
    be = np.sum(results) / np.prod(dvf.shape[:-1])
    logger.info(f"Bending Energy: {be}")
    return be


def dice_similarity(moving_deformed, fixed, levels):
    thresholds_moving = threshold_multiotsu(moving_deformed, classes=levels)
    thresholds_fixed = threshold_multiotsu(fixed, classes=levels)
    regions_moving_deformed = np.digitize(moving_deformed, bins=thresholds_moving)
    regions_fixed = np.digitize(fixed, bins=thresholds_fixed)

    similarities = []
    for region_value in np.unique(regions_fixed)[1:]:
        intersection = np.sum(
            (regions_moving_deformed == regions_fixed) & (regions_fixed == region_value)
        )
        sum_pixels = np.sum(regions_fixed == region_value) + np.sum(
            regions_moving_deformed == region_value
        )
        similarities.append(2.0 * intersection / sum_pixels)

    logger.info(f"Dice Similarities: {similarities}")
    return similarities


def hausdorff_distance(surface_points, surface_points_deformed, spacing=1):
    distances = []
    for i in range(len(surface_points)):
        max_distance1 = (
            np.max(distance.cdist(surface_points[i], surface_points_deformed[i]).min(axis=1))
            * spacing
        )
        max_distance2 = (
            np.max(distance.cdist(surface_points_deformed[i], surface_points[i]).min(axis=1))
            * spacing
        )
        distances.append(max(max_distance1, max_distance2))
    logger.info(f"Hausdorff Distances: {distances}")
    return distances


def dvf_rmse(dvf1, dvf2, spacing=1):
    rmse = np.linalg.norm((dvf1 - dvf2) * spacing, axis=3).mean()
    logger.info(f"DVF RMSE: {rmse}")
    return rmse


def mean_surface_distance(surface_points, surface_points_deformed, spacing=1):
    distances = []
    for i in range(len(surface_points)):
        mean_distance1 = (
            np.mean(distance.cdist(surface_points[i], surface_points_deformed[i]).min(axis=1))
            * spacing
        )
        mean_distance2 = (
            np.mean(distance.cdist(surface_points_deformed[i], surface_points[i]).min(axis=1))
            * spacing
        )
        distances.append((mean_distance1 + mean_distance2) / 2)
    logger.info(f"Mean Surface Distances: {distances}")
    return distances


def tre(lms1, lms2, spacing=1):
    tre = np.linalg.norm((lms1 - lms2) * spacing, axis=1).mean()
    logger.info(f"TRE: {tre}")
    return tre


def tre_hist(lms1, lms2, spacing=1, ax=None, bins=40):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    tres = np.linalg.norm((lms1 - lms2) * spacing, axis=1)
    ax.hist(tres, bins=bins)
    ax.set_xlabel("TRE (mm)")
    ax.set_ylabel("Count")
    return fig


def jacobian_determinant(dvf, ax=None, vmin=None, vmax=None, plot=True):
    axis_swap = 2 if len(dvf.shape) == 4 else 1
    dvf = np.swapaxes(dvf, 0, axis_swap)
    dvf_img = sitk.GetImageFromArray(dvf, isVector=True)
    jac_det_contracting_field = sitk.DisplacementFieldJacobianDeterminant(dvf_img)
    jac_det = sitk.GetArrayFromImage(jac_det_contracting_field)
    jac_det = np.swapaxes(jac_det, 0, axis_swap)
    if len(jac_det.shape) == 3:
        slice = jac_det.shape[-1] // 2
        jac_det = jac_det[..., slice]

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(
            jac_det,
            cmap="jet",
            ax=ax,
            square=True,
            cbar_kws={"fraction": 0.045, "pad": 0.02},
            vmin=vmin,
            vmax=vmax,
        )
        ax.invert_xaxis()
        ax.set_xticks([])
        ax.set_yticks([])

    logger.info(f"Jacobian min,max: {np.min(jac_det)}, {np.max(jac_det)}")
    return jac_det


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
    ax=None,
):
    if len(data.shape) == 2:
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(data, cmap="gray")
        return

    if y_slice_depth is None:
        y_slice_depth = data.shape[1] // 2

    if ax is None:
        ax = plt.figure(figsize=(8, 8)).add_subplot(projection="3d")

    sliced_data = np.copy(data)
    sliced_data[:, :y_slice_depth, :] = 0
    sliced_data[sliced_data < 10] = 0

    cmap = cm.get_cmap(cmap_name)
    norm = Normalize(vmin=np.min(sliced_data), vmax=1.5 * np.max(sliced_data))

    colors = np.array(list(map(lambda x: get_cmap_color(cmap, norm(x), alpha), sliced_data)))

    ax.voxels(sliced_data, facecolors=colors, edgecolor=(0, 0, 0, 0.2))
    ax.set_xlim3d(0, data.shape[0] + 2)
    ax.set_ylim3d(5, data.shape[1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.minorticks_off()
    ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
    ax.locator_params(axis="y", nbins=3)
    ax.view_init(*orientation)

    if lms is not None:
        for lm in lms:
            if abs(lm[1] - y_slice_depth) <= 0.25:
                ax.scatter(lm[0], lm[1], lm[2], s=50, c="r")


def plot_dvf(
    data, scale=1, invert=False, slice=None, ax=None, vmin=None, vmax=None, max_vectors=None
):
    if len(data.shape[:-1]) > 2:
        if slice is None:
            slice = data.shape[2] // 2
        data = data[:, :, slice, :]

    if max_vectors is not None:
        factors = [min(1.0, max_vectors / x) for x in data.shape[:-1]] + [1.0]
        data = zoom(data, factors)

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

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    qq = ax.quiver(
        X,
        Y,
        v,
        u,
        c,
        scale=scale,
        units="xy",
        angles="xy",
        scale_units="xy",
        cmap=plt.cm.jet,
        clim=(vmin, vmax),
    )

    ax.set_xlim(0, data.shape[0])
    ax.set_ylim(0, data.shape[1])
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().colorbar(qq, fraction=0.045, pad=0.02, label="Displacement magnitude", ax=ax)
    return fig


def plot_cpoints(
    points, grid_spacing, grid_origin, grid_direction, slice=None, ax=None, colors=None, alpha=0.3
):
    points_slice = points

    if len(points.shape) == 4:
        if slice is None:
            slice = points.shape[2] // 2
        points_slice = points[:, :, slice, :]
        points_slice = np.swapaxes(points_slice, 0, 1)

    grid_spacing = np.array(grid_spacing)
    grid_origin = np.array(grid_origin)
    grid_origin = grid_origin + 0.5

    X, Y = np.meshgrid(
        *[
            np.arange(
                grid_origin[i],
                grid_origin[i] + grid_direction[i, i] * grid_spacing[i] * points_slice.shape[i],
                grid_spacing[i] * grid_direction[i, i],
            )
            for i in range(len(points_slice.shape[:-1]))
        ],
        indexing="xy",
    )

    cmap = None

    if colors is None:
        colors = np.zeros(points_slice.shape[:-1])
        x_slice = colors.shape[0] // 2
        y_slice = colors.shape[1] // 2
        colors[:x_slice, :y_slice] = 0.25
        colors[x_slice:, :y_slice] = 0.5
        colors[:x_slice, y_slice:] = 0.75
        colors[x_slice:, y_slice:] = 1.0

        colormap_colors = ["#ffcc00", "red", "green", "blue"]
        cmap = LinearSegmentedColormap.from_list("quadrants", colormap_colors)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(X, Y, marker="+", c=colors, cmap=cmap, alpha=alpha, s=20)

    ax.grid(False)
    ax.scatter(
        points_slice[..., 0],
        points_slice[..., 1],
        marker="s",
        s=15,
        c=colors,
        cmap=cmap,
        alpha=0.8,
    )


def cpoint_cloud(points):
    max_level = points.shape[0] - 1
    point_cloud = []

    for index in np.ndindex(points.shape[:-1]):
        point = points[index]
        min_index = np.min(index)
        max_index = np.max(index)
        level = min_index if min_index < max_level - max_index else max_level - max_index
        point_cloud.append([*point, level])

    return np.array(point_cloud)


def plot_color_diff(moving, source, aspect, slice_tuple, invert_y=True, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    img1 = sitk.GetImageFromArray(moving[slice_tuple])
    img2 = sitk.GetImageFromArray(source[slice_tuple])
    img_min = np.min([img1, img2])
    img_max = np.max([img1, img2])

    img1_255 = sitk.Cast(
        sitk.IntensityWindowing(
            img1,
            windowMinimum=img_min,
            windowMaximum=img_max,
            outputMinimum=0.0,
            outputMaximum=255.0,
        ),
        sitk.sitkUInt8,
    )
    img2_255 = sitk.Cast(
        sitk.IntensityWindowing(
            img2,
            windowMinimum=img_min,
            windowMaximum=img_max,
            outputMinimum=0.0,
            outputMaximum=255.0,
        ),
        sitk.sitkUInt8,
    )

    img3 = sitk.Cast(sitk.Compose(img1_255, img2_255, img1_255), sitk.sitkVectorUInt8)
    arr = sitk.GetArrayFromImage(img3)

    ax.imshow(np.swapaxes(arr, 0, 1), aspect=aspect)
    if invert_y:
        ax.invert_yaxis()
    else:
        ax.invert_xaxis()
    ax.set_ylim(30, 180)
    ax.axis("off")
    return ax.get_figure()


def calc_validation(result: RunResult):
    logger.info(f"Calculating validation metrics for {result.instance.collection}:")
    start = time.perf_counter()

    metrics = validation_metrics(result)
    metrics.extend(validation_visualization(result))

    end = time.perf_counter()
    logger.info(f"Validation metrics calculated in {end - start:.2f}s")

    return metrics


def validation_metrics(result: RunResult):
    metrics = []

    metrics.append(
        {
            "validation/tre": tre(
                result.deformed_lms, result.instance.lms_moving, result.instance.spacing
            )
        }
    )

    if result.instance.collection == Collection.SYNTHETIC:
        dvf_copy = np.copy(result.dvf)
        mask = np.linalg.norm(result.instance.dvf, axis=-1) > 0
        dvf_copy[~mask] = np.array([0 for _ in range(dvf_copy.shape[-1])])
        metrics.append({"validation/dvf_rmse": dvf_rmse(dvf_copy, result.instance.dvf)})
        hd_dists = hausdorff_distance(
            result.instance.surface_points, result.deformed_surface_points
        )
        md_dists = mean_surface_distance(
            result.instance.surface_points, result.deformed_surface_points
        )
        metrics.append({"validation/hausdorff_cube": hd_dists[0]})
        metrics.append({"validation/hausdorff_sphere": hd_dists[1]})
        metrics.append({"validation/mean_surface_cube": md_dists[0]})
        metrics.append({"validation/mean_surface_sphere": md_dists[1]})
        metrics.append({"validation/bending_energy": bending_energy(result.dvf)})

    return metrics


def validation_visualization(result: RunResult, clim_dvf=(None, None), clim_jac=(None, None)):
    figs = [] # aggregate all figures
    instance = result.instance
    collection = instance.collection
    to_dict = lambda x, title: {f"visualization/{title}": x}

    # Visualizations for either SYNTHETIC or EXAMPLES results:
    if collection == Collection.SYNTHETIC or collection == Collection.EXAMPLES:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        slice_txt = "" if collection == Collection.EXAMPLES else "(slice)"
        if collection == collection.SYNTHETIC:
            axes[0, 0].remove()
            axes[0, 0] = fig.add_subplot(2, 2, 1, projection="3d")
        plot_voxels(result.deformed, ax=axes[0, 0])
        plot_cpoints(
            result.control_points,
            result.grid_spacing,
            result.grid_origin,
            instance.direction,
            ax=axes[0, 1],
        )
        plot_dvf(result.dvf, ax=axes[1, 0], vmin=clim_dvf[0], vmax=clim_dvf[1])
        jacobian_determinant(result.dvf, ax=axes[1, 1], vmin=clim_jac[0], vmax=clim_jac[1])
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        axes[1, 0].set_title(f"DVF {slice_txt}", fontsize=12)
        axes[0, 0].set_title(f"Deformed source", fontsize=12)
        axes[1, 1].set_title(f"Jacobian determinant {slice_txt}", fontsize=12)
        axes[0, 1].set_title(f"Control points {slice_txt}", fontsize=12)
        figs.append(to_dict(wandb.Image(fig), "overview"))
    
    # Visualizations for LEARN results:
    elif collection == Collection.LEARN:
        figs.append(to_dict(wandb.Object3D(cpoint_cloud(result.control_points)), "cpoints"))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plot_color_diff(
            result.deformed, instance.fixed, 1.4, (50, slice(None), slice(None)), ax=axes[0]
        )
        plot_color_diff(
            result.deformed,
            instance.fixed,
            1.0,
            (slice(None), 50, slice(None)),
            invert_y=False,
            ax=axes[1],
        )
        fig.tight_layout()
        figs.append(to_dict(wandb.Image(fig), "overview"))

    # TRE distribution plot
    tre_img = wandb.Image(tre_hist(result.deformed_lms, instance.lms_moving, instance.spacing))
    figs.append(to_dict(tre_img, "tre_hist"))

    return figs


def get_vmin_vmax(result1: RunResult, result2: RunResult):
    result1_mag = np.sqrt(
        result1.dvf[:, :, result1.dvf.shape[2] // 2, 0] ** 2
        + result1.dvf[:, :, result1.dvf.shape[2] // 2, 1] ** 2
    )
    result2_mag = np.sqrt(
        result2.dvf[:, :, result2.dvf.shape[2] // 2, 0] ** 2
        + result2.dvf[:, :, result2.dvf.shape[2] // 2, 1] ** 2
    )
    vmin_dvf = np.min([np.min(result1_mag), np.min(result2_mag)])
    vmax_dvf = np.max([np.max(result1_mag), np.max(result2_mag)])

    jac_hybrid = jacobian_determinant(result1.dvf, plot=False)
    jac_baseline = jacobian_determinant(result2.dvf, plot=False)
    vmin_jac = np.min([np.min(jac_baseline), np.min(jac_hybrid)])
    vmax_jac = np.max([np.max(jac_baseline), np.max(jac_hybrid)])

    return (vmin_dvf, vmax_dvf), (vmin_jac, vmax_jac)
