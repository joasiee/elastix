from pathlib import Path
import logging
import time
import re
from tempfile import TemporaryDirectory

from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.spatial import distance
from skimage.filters import threshold_multiotsu
import numpy as np
import SimpleITK as sitk
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.interpolate import RegularGridInterpolator

from thesispy.elastix_wrapper.wrapper import generate_transformed_points
from thesispy.experiments.instance import RunResult
from thesispy.definitions import N_CORES, Collection
from thesispy.colorline import colorline

logger = logging.getLogger("Validation")
VALIDATION_NAMES_NEW = [
    "tre",
    "mean_surface_cube",
    "mean_surface_sphere",
    "dice_sphere",
    "bending_energy",
    "dvf_rmse",
]
VALIDATION_ABBRVS_NEW = [
    "$TRE$",
    "$ASD_{\\textsc{cube}}$",
    "$ASD_{\\textsc{sphere}}$",
    "$DSC_{\\textsc{sphere}}$",
    "$E_b$",
    "$\\vec{v}_{\\epsilon}$",
]

INV_MAPPING = {0: 2, 1: 1, 2: 0}


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
    return p, sum


def bending_energy(dvf, downscaling_f: int = 1, mask=None, use_tqdm: bool = True, sum: bool = True):
    """Computes the bending energy of a given DVF.
    
    Args:
        dvf (np.ndarray): The DVF to compute the bending energy for.
        downscaling_f (int, optional): The factor by which the DVF is downsampled. Defaults to 1.
        mask (np.ndarray, optional): Only calculate the bending energy for voxels within this mask. Defaults to None.
        use_tqdm (bool, optional): Whether to use tqdm to display the progress. Defaults to True.
        sum (bool, optional): Whether to return the sum of the bending energy or a map with for each voxel the bending energy. Defaults to True.
    """
    if mask is not None:
        mask = mask[::downscaling_f, ::downscaling_f, ::downscaling_f]

    dvf = dvf[::downscaling_f, ::downscaling_f, ::downscaling_f, :]
    nr_points = np.prod(dvf.shape[:-1]) if mask is None else np.sum(mask)

    results = ProgressParallel(
        n_jobs=N_CORES, desc="computing bending energy", backend="multiprocessing", total=nr_points, use_tqdm=use_tqdm
    )(delayed(bending_energy_point)(dvf, p) for p in np.ndindex(dvf.shape[:-1]) if mask is None or mask[p])

    be_map = np.zeros(dvf.shape[:-1])
    for p, be in results:
        be_map[p] = be

    return be_map.sum() if sum else be_map


def dice_similarity(moving_deformed, fixed, levels):
    """Computes the Dice Similarity Coefficient using Otsu multi-level thresholding."""
    thresholds_moving = threshold_multiotsu(moving_deformed, classes=levels)
    thresholds_fixed = threshold_multiotsu(fixed, classes=levels)
    regions_moving_deformed = np.digitize(moving_deformed, bins=thresholds_moving)
    regions_fixed = np.digitize(fixed, bins=thresholds_fixed)

    similarities = []
    for region_value in np.unique(regions_fixed)[1:]:
        similarities.append(dice_similarity_(regions_moving_deformed, regions_fixed, region_value))

    logger.info(f"Dice Similarities: {similarities}")
    return similarities


def dice_similarity_(region_a, region_b, value):
    intersection = np.sum((region_a == region_b) & (region_b == value))
    sum_pixels = np.sum(region_a == value) + np.sum(region_b == value)
    return 2.0 * intersection / sum_pixels


def hausdorff_distance(surface_points, surface_points_deformed, spacing=1):
    """Computes the Hausdorff Distance between two sets of points."""
    distances = []
    for i in range(len(surface_points)):
        max_distance1 = np.max(distance.cdist(surface_points[i], surface_points_deformed[i]).min(axis=1)) * spacing
        max_distance2 = np.max(distance.cdist(surface_points_deformed[i], surface_points[i]).min(axis=1)) * spacing
        distances.append(max(max_distance1, max_distance2))
    logger.info(f"Hausdorff Distances: {distances}")
    return distances


def dvf_rmse(dvf1, dvf2, spacing=1):
    """Computes the RMSE between two DVF's."""
    rmse = np.linalg.norm((dvf1 - dvf2) * spacing, axis=3).mean()
    logger.info(f"DVF RMSE: {rmse}")
    return rmse


def mean_surface_distance(surface_points, surface_points_deformed, spacing=1):
    """Computes the Mean Surface Distance between two sets of points."""
    distances = []
    for i in range(len(surface_points)):
        mean_distance1 = np.mean(distance.cdist(surface_points[i], surface_points_deformed[i]).min(axis=1)) * spacing
        mean_distance2 = np.mean(distance.cdist(surface_points_deformed[i], surface_points[i]).min(axis=1)) * spacing
        distances.append((mean_distance1 + mean_distance2) / 2)
    logger.info(f"Mean Surface Distances: {distances}")
    return distances


def tre(result: RunResult):
    """Computes the TRE between the deformed landmarks and the ground truth landmarks."""
    tre = np.linalg.norm((result.deformed_lms - result.instance.lms_moving) * result.instance.spacing, axis=1).mean()
    logger.info(f"TRE: {tre}")
    return tre


def tre_hist(lms1, lms2, spacing=1, ax=None, bins=40):
    """Plots a histogram of the TRE between two sets of landmarks."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    tres = np.linalg.norm((lms1 - lms2) * spacing, axis=1)
    ax.hist(tres, bins=bins)
    ax.set_xlabel("TRE (mm)")
    ax.set_ylabel("Count")
    return ax.get_figure()


def tre_hist_wandb(result: RunResult, bins=40):
    tres = np.linalg.norm((result.deformed_lms - result.instance.lms_moving) * result.instance.spacing, axis=1)
    return wandb.Histogram(tres, num_bins=bins)


def jacobian_determinant(dvf, ax=None, vmin=None, vmax=None, plot=True):
    """Calculates the Jacobian determinant of a DVF and plots it as a heatmap.
    
    Args:
        dvf: The DVF.
        ax: The axis to plot on.
        vmin: The minimum value for the colorbar.
        vmax: The maximum value for the colorbar.
        plot: Whether to plot the heatmap.
    """
    dvf_img = sitk.GetImageFromArray(dvf, isVector=True)
    jac_det_contracting_field = sitk.DisplacementFieldJacobianDeterminant(dvf_img)
    jac_det = sitk.GetArrayFromImage(jac_det_contracting_field)

    jac_det_slice = jac_det
    if len(jac_det.shape) == 3:
        slice = jac_det.shape[0] // 2
        jac_det_slice = jac_det[slice, :, :]

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(
            jac_det_slice,
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
    return jac_det_slice, jac_det


def jacobian_determinant_masked(run_result: RunResult, slice_tuple, ax=None):
    """Plots the voxels of the fixed image within the mask with the Jacobian determinant as color.
    
    Args:
        run_result: A run result.
        slice_tuple: The slice to plot.
        ax: The axis to plot on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    dvf_img = sitk.GetImageFromArray(run_result.dvf, isVector=True)
    jac_det_contracting_field = sitk.DisplacementFieldJacobianDeterminant(dvf_img)
    jac_det = sitk.GetArrayFromImage(jac_det_contracting_field)

    mask = run_result.instance.mask
    fixed = run_result.instance.fixed
    origin = run_result.instance.origin
    spacing = run_result.instance.spacing
    size = run_result.instance.size
    indices_xy = get_indices_xy(slice_tuple)

    mask_slice = mask[slice_tuple]
    max_indices = np.where(mask_slice == 1)
    margin = 5
    min_x, max_x = np.min(max_indices[1]) - margin, np.max(max_indices[1]) + margin
    min_x *= spacing[indices_xy[0]]
    max_x *= spacing[indices_xy[0]]
    min_y, max_y = np.min(max_indices[0]) - margin, np.max(max_indices[0]) + margin
    min_y *= spacing[indices_xy[1]]
    max_y *= spacing[indices_xy[1]]

    fixed_slice = np.copy(fixed[slice_tuple])
    fixed_slice[mask_slice == 0] = np.nan
    gray_cmap = plt.cm.get_cmap("gray")
    gray_cmap.set_bad(alpha=0)

    jac_det_slice = jac_det[slice_tuple]
    jac_det_slice[mask_slice == 0] = np.nan
    jet_cmap = plt.cm.get_cmap("jet")
    jet_cmap.set_bad(alpha=0)

    extent = (
        origin[indices_xy[0]],
        size[indices_xy[0]] * spacing[indices_xy[0]],
        size[indices_xy[1]] * spacing[indices_xy[1]],
        origin[indices_xy[1]],
    )

    ax.imshow(fixed_slice, cmap="gray", alpha=0.6, extent=extent)
    im_jac = ax.imshow(jac_det_slice, alpha=0.5, cmap="jet", extent=extent)
    cbar = ax.get_figure().colorbar(im_jac, ax=ax, location="bottom", pad=0.1, alpha=1.0)
    cbar.set_label("Contraction --> Expansion")

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    if slice_tuple[1] != slice(None, None, None) or slice_tuple[0] != slice(None, None, None):
        ax.invert_xaxis()

    ax.axis("off")
    return ax.get_figure()


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
    """Plots a 3D volume as voxels.
    
    Args:
        data: The data to plot.
        lms: The landmarks to plot (untested).
        y_slice_depth: The depth of the slice to plot.
        orientation: The orientation of the plot.
        cmap_name: The colormap to use.
        alpha: The alpha value of the voxels.
        ax: The axis to plot on.
    """
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

    ax.voxels(sliced_data, facecolors=colors, edgecolor=(0, 0, 0, 0.1))
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


def plot_dvf(data, scale=1, invert=False, slice=None, ax=None, vmin=None, vmax=None):
    """Plots a 2D or 3D DVF as a quiver plot.
    
    Args:
        data: The DVF to plot.
        scale: The scale of the arrows.
        invert: Whether to invert the DVF.
        slice: The slice to plot.
        ax: The axis to plot on.
        vmin: The minimum value of the colormap.
        vmax: The maximum value of the colormap.
    """
    if len(data.shape[:-1]) > 2:
        if slice is None:
            slice = data.shape[0] // 2
        data = data[slice, :, :, :]

    X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    if invert:
        X = X + data[..., 0]
        Y = Y + data[..., 1]
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
        u,
        v,
        c,
        scale=scale,
        units="xy",
        angles="xy",
        scale_units="xy",
        cmap=plt.cm.jet,
        clim=(vmin, vmax),
    )

    ax.set_xlim(0, data.shape[1])
    ax.set_ylim(0, data.shape[0])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().colorbar(qq, fraction=0.045, pad=0.02, label="Displacement magnitude", ax=ax)
    return ax.get_figure()


def plot_cpoints(run_result: RunResult, slice_index=None, slice_pos=1, ax=None, colors=None, alpha=0.3, marker_size=20):
    """Plots the control points of a run result."""
    points = run_result.control_points
    points_slice = points

    grid_spacing = np.array(run_result.grid_spacing)
    grid_origin = np.array(run_result.grid_origin)
    grid_direction = run_result.instance.direction.reshape(len(grid_origin), len(grid_origin))
    grid_origin = grid_direction @ grid_origin

    slice_tuple = [slice(None), slice(None), slice(None)]
    if len(points.shape) == 4:
        if slice_index is None:
            slice_index = points.shape[2] // 2
        slice_tuple[slice_pos] = slice_index
        points_slice = points[tuple(slice_tuple)]

    indices_xy = get_indices_xy(slice_tuple, inv=False)

    for p in np.ndindex(points_slice.shape[:-1]):
        points_slice[p] = grid_direction @ points_slice[p]

    X, Y = mesh_grids(grid_origin, grid_spacing, indices_xy, points.shape[:-1], 2, indexing="ij")

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

    ax.scatter(X, Y, marker="+", c=colors, cmap=cmap, alpha=alpha, s=marker_size)

    ax.grid(False)
    ax.scatter(
        points_slice[..., indices_xy[0]],
        points_slice[..., indices_xy[1]],
        marker="s",
        s=marker_size,
        c=colors,
        cmap=cmap,
        alpha=0.8,
    )


def cpoint_cloud(points):
    """Converts a control point grid to a point cloud to store in WandB."""
    max_level = points.shape[0] - 1
    point_cloud = []

    for index in np.ndindex(points.shape[:-1]):
        point = points[index]
        min_index = np.min(index)
        max_index = np.max(index)
        level = min_index if min_index < max_level - max_index else max_level - max_index
        point_cloud.append([*point, level])

    return np.array(point_cloud)


def plot_color_diff(result: RunResult, slice_tuple, ax=None):
    """Plots the difference between the fixed and deformed image as a color image."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    moving = result.deformed
    target = result.instance.fixed

    img1 = sitk.GetImageFromArray(moving[slice_tuple])
    img2 = sitk.GetImageFromArray(target[slice_tuple])
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

    spacing = result.instance.spacing
    origin = result.instance.origin
    size = result.instance.size

    indices_xy = get_indices_xy(slice_tuple)
    extent = (
        origin[indices_xy[0]],
        size[indices_xy[0]] * spacing[indices_xy[0]],
        size[indices_xy[1]] * spacing[indices_xy[1]],
        origin[indices_xy[1]],
    )

    invert_x = slice_tuple[1] != slice(None, None, None); invert_y = slice_tuple[0] != slice(None, None, None)
    ylims = None
    if slice_tuple[1] != slice(None, None, None) or slice_tuple[2] != slice(None, None, None):
        ylims = (80, 300)

    return plot_image(arr, extent=extent, inverts=(invert_x, invert_y), ylims=ylims, ax=ax, cmap=None)


def plot_image(image: np.ndarray, extent=None, inverts=(False, False), ylims=None, ax=None, cmap="gray"):
    """Plots an image with the correct orientation and extent."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(image, extent=extent, cmap=cmap)

    invert_x, invert_y = inverts
    if invert_x or invert_y:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()
    if ylims is not None:
        ax.set_ylim(ylims[0], ylims[1])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_linewidth(2)
    return ax.get_figure()


def plot_dvf_masked(run_result, slice_tuple, ax=None, zoom_f=5):
    """Plots the DVF with as background the fixed image."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    fixed = run_result.instance.fixed
    dvf = run_result.dvf
    spacing = run_result.instance.spacing
    origin = run_result.instance.origin
    size = run_result.instance.size
    mask = run_result.instance.mask
    direction = run_result.instance.direction

    image_slice = fixed[slice_tuple]

    indices_xy = get_indices_xy(slice_tuple)
    extent = (
        origin[indices_xy[0]],
        size[indices_xy[0]] * spacing[indices_xy[0]],
        size[indices_xy[1]] * spacing[indices_xy[1]],
        origin[indices_xy[1]],
    )

    t = ax.imshow(image_slice, extent=extent)
    ax.invert_yaxis()
    if slice_tuple[1] != slice(None, None, None) or slice_tuple[0] != slice(None, None, None):
        ax.invert_xaxis()
    t.set_cmap("gray")

    df_slice = np.copy(dvf[slice_tuple + (slice(None),)])
    mask_slice = mask[slice_tuple]
    df_slice[mask_slice == 0] = np.nan

    direction = np.array(direction).reshape((3, 3))
    voxel_to_physical = direction
    physical_to_voxel = np.linalg.inv(voxel_to_physical)

    for i in range(df_slice.shape[0]):
        for j in range(df_slice.shape[1]):
            p = physical_to_voxel @ df_slice[i, j]
            df_slice[i, j] = p

    coordsX, coordsY = mesh_grids(origin, spacing, indices_xy, size, 2)

    x = df_slice[:, :, indices_xy[0]]
    y = df_slice[:, :, indices_xy[1]]
    coordsX = coordsX[::zoom_f, ::zoom_f]
    coordsY = coordsY[::zoom_f, ::zoom_f]
    x = x[::zoom_f, ::zoom_f]
    y = y[::zoom_f, ::zoom_f]

    M = np.sqrt(x * x + y * y)
    qq = ax.quiver(
        coordsX,
        coordsY,
        x,
        y,
        M,
        cmap=plt.cm.jet,
        scale_units="xy",
        angles="xy",
        scale=1,
        minlength=0,
        headaxislength=5.0,
        width=0.003,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_linewidth(2)
    if slice_tuple[2] != slice(None, None, None) or slice_tuple[1] != slice(None, None, None):
        ax.set_ylim(80, 300)
    return ax.get_figure()


def plot_dvf_3d(run_result, zoom_f=6, ax=None):
    """Plots the DVF in 3D."""
    if ax is None:
        ax = plt.figure(figsize=(5, 5)).add_subplot(projection="3d")
    dvf = np.copy(run_result.dvf)
    mask = run_result.instance.mask
    dvf[mask == 0] = np.nan
    dvf = np.swapaxes(dvf, 0, 2)

    spacing = run_result.instance.spacing
    origin = run_result.instance.origin
    direction = run_result.instance.direction

    direction = np.array(direction).reshape((3, 3))
    voxel_to_physical = direction
    physical_to_voxel = np.linalg.inv(voxel_to_physical)

    sizes = np.copy(dvf.shape[:-1])
    dvf = dvf[::zoom_f, ::zoom_f, ::zoom_f, :]

    for p in np.ndindex(dvf.shape[:3]):
        dvf[p] = physical_to_voxel @ dvf[p]

    x, y, z = dvf[:, :, :, 0], dvf[:, :, :, 1], dvf[:, :, :, 2]

    X, Y, Z = mesh_grids(origin, spacing, [0, 1, 2], sizes, 3, indexing="ij")
    X, Y, Z = (
        X[::zoom_f, ::zoom_f, ::zoom_f],
        Y[::zoom_f, ::zoom_f, ::zoom_f],
        Z[::zoom_f, ::zoom_f, ::zoom_f],
    )

    M = np.sqrt(x * x + y * y + z * z)
    M = M[~np.isnan(M)]

    q = ax.quiver(X, Y, Z, x, y, z, cmap="jet", linewidths=0.6)
    q.set_array(M.flatten())

    ax.view_init(20, -60)

    ax.set_xlim3d(10, sizes[0] * spacing[0] - 30)
    ax.set_ylim3d(0, sizes[1] * spacing[1] - 40)
    ax.set_zlim3d(0, sizes[2] * spacing[2])

    ax.set_xlabel("front view")
    ax.set_ylabel("side view")
    ax.set_zlabel("vertical view", labelpad=0)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    return ax.get_figure()


def generate_mesh_lines_points(
    origin, spacing, size, direction, step_size, slice_tuple, nr_line_points=1000, mask=None
):
    indices_xy = get_indices_xy(slice_tuple)
    i_x, i_y = indices_xy[0], indices_xy[1]
    i_slice = [i for i in range(3) if i not in indices_xy][0]
    slice_val = [i for i in slice_tuple if not isinstance(i, slice)][0]

    min_x = 0
    max_x = size[i_x]
    min_y = 0
    max_y = size[i_y]

    if mask is not None:
        mask_slice = mask[slice_tuple]
        margin = 10
        min_x = max(np.min(np.where(np.any(mask_slice, axis=0))[0]) - margin, 0)
        max_x = min(np.max(np.where(np.any(mask_slice, axis=0))[0]) + margin, size[i_x])
        min_y = max(np.min(np.where(np.any(mask_slice, axis=1))[0]) - margin, 0)
        max_y = min(np.max(np.where(np.any(mask_slice, axis=1))[0]) + margin, size[i_y])

    xs = np.arange(min_x, max_x, step_size)
    ys = np.arange(min_y, max_y, step_size)

    lines_x = []
    lines_y = []

    for x in xs:
        y_vals = np.linspace(min_y, max_y, nr_line_points)
        line = np.zeros((nr_line_points, 3))
        line[:, i_x] = x * spacing[i_x] * direction[i_x][i_x] + origin[i_x]
        line[:, i_y] = y_vals * spacing[i_y] * direction[i_y][i_y] + origin[i_y]
        line[:, i_slice] = slice_val * spacing[i_slice] * direction[i_slice][i_slice] + origin[i_slice]

        lines_x.append(line)

    lines_x.pop(0)

    for y in ys:
        x_vals = np.linspace(min_x, max_x, nr_line_points)
        line = np.zeros((nr_line_points, 3))
        line[:, i_x] = x_vals * spacing[i_x] * direction[i_x][i_x] + origin[i_x]
        line[:, i_y] = y * spacing[i_y] * direction[i_y][i_y] + origin[i_y]
        line[:, i_slice] = slice_val * spacing[i_slice] * direction[i_slice][i_slice] + origin[i_slice]

        lines_y.append(line)

    lines_y.pop(0)

    return lines_x + lines_y


def write_line_as_points(line, filename: Path):
    with open(filename, "w") as f:
        f.write("point\n")
        f.write(f"{len(line)}\n")
        for point in line:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")


def read_deformed_line(filename: Path):
    with open(filename, "r") as f:
        lines = f.readlines()
        nr_points = len(lines)
        points = np.zeros((nr_points, 3))
        for i, line in enumerate(lines):
            match = re.search(r"OutputPoint = \[ (.*) (.*) (.*) \]\t", line)
            points[i, 0] = float(match.group(1))
            points[i, 1] = float(match.group(2))
            points[i, 2] = float(match.group(3))

    return points


def plot_deformed_mesh(result: RunResult, slice_tuple, nr_lines=12, nr_line_points=1000, ax=None, fix_axes=False, vmin=None, vmax=None):
    """Plots the deformed mesh in the given slice overlaid on the fixed image."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    origin = result.instance.origin
    spacing = result.instance.spacing
    size = result.instance.size
    direction = result.instance.direction
    transform_params = result.transform_params
    step_size = max(size) // nr_lines

    with TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)

        lines = generate_mesh_lines_points(
            origin, spacing, size, direction, step_size, slice_tuple, nr_line_points, mask=result.instance.mask
        )
        deformed_lines = []
        for _, line in enumerate(lines):
            write_line_as_points(line, out_dir / f"line.txt")
            generate_transformed_points(transform_params, out_dir, out_dir / f"line.txt")
            deformed_lines.append(read_deformed_line(out_dir / f"outputpoints.txt"))

    fixed_image = result.instance.fixed
    fixed_image_slice = fixed_image[slice_tuple]
    i_x, i_y = get_indices_xy(slice_tuple)
    extent = [
        origin[i_x],
        origin[i_x] + size[i_x] * spacing[i_x] * direction[i_x][i_x],
        origin[i_y] + size[i_y] * spacing[i_y] * direction[i_y][i_y],
        origin[i_y],
    ]

    ax.imshow(fixed_image_slice, cmap="gray", extent=extent)
    ax.set_xlim(extent[0], extent[1])

    if fix_axes:
        ax.invert_yaxis()
        if slice_tuple[1] != slice(None, None, None) or slice_tuple[0] != slice(None, None, None):
            ax.invert_xaxis()
        if result.instance.instance == 1 and slice_tuple[2] != slice(None, None, None) or slice_tuple[1] != slice(None, None, None):
            ax.set_ylim(80, 300)

    voxel_to_physical = (spacing * np.identity(3)) @ direction
    physical_to_voxel = np.linalg.inv(voxel_to_physical)
    fn = RegularGridInterpolator(
        (np.arange(0, size[2]), np.arange(0, size[1]), np.arange(0, size[0])),
        result.dvf,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )
    color_lists = []

    for line in lines:
        indices = np.flip(np.array([physical_to_voxel @ point for point in line]), axis=1)
        displacements = fn(indices)
        colors = np.sqrt(displacements[:, i_x] ** 2 + displacements[:, i_y] ** 2)
        color_lists.append(colors)
        ax.plot(line[:, i_x], line[:, i_y], color="white", linewidth=0.5, linestyle="dotted", alpha=0.5)

    if vmin is None or vmax is None:
        vmin = np.min([np.min(color_list) for color_list in color_lists])
        vmax = np.max([np.max(color_list) for color_list in color_lists])
    
    norm = plt.Normalize(vmin, vmax)

    for i, line in enumerate(deformed_lines):
        colorline(line[:, i_x], line[:, i_y], z=color_lists[i], norm=norm, ax=ax, cmap="jet", linewidth=1, alpha=1)

    ax.axis("off")

    return ax.get_figure(), (vmin, vmax)


def calc_validation(result: RunResult):
    """Perform all validation calculations and visualizations."""
    logger.info(f"Calculating validation metrics for {result.instance.collection}:")
    start = time.perf_counter()

    metrics = validation_metrics(result)
    metrics.extend(validation_visualization(result))

    end = time.perf_counter()
    logger.info(f"Validation metrics calculated in {end - start:.2f}s")

    return metrics


def validation_metrics(result: RunResult):
    """Calculate all validation metrics."""
    metrics = []

    metrics.append({"validation/tre": tre(result)})
    metrics.append({"validation/bending_energy": result.bending_energy})
    logger.info(f"Bending Energy: {result.bending_energy:.2f}")

    if result.instance.collection == Collection.SYNTHETIC:
        dvf_copy = np.copy(result.dvf)
        mask = np.linalg.norm(result.instance.dvf, axis=-1) > 0
        dvf_copy[~mask] = np.array([0 for _ in range(dvf_copy.shape[-1])])
        metrics.append({"validation/dvf_rmse": dvf_rmse(dvf_copy, result.instance.dvf)})
        hd_dists = hausdorff_distance(result.instance.surface_points, result.deformed_surface_points)
        md_dists = mean_surface_distance(result.instance.surface_points, result.deformed_surface_points)
        dsc_sims = dice_similarity(result.deformed, result.instance.fixed, 3)
        bending_energy_crude = bending_energy(result.dvf)
        metrics.append({"validation/hausdorff_cube": hd_dists[0]})
        metrics.append({"validation/hausdorff_sphere": hd_dists[1]})
        metrics.append({"validation/mean_surface_cube": md_dists[0]})
        metrics.append({"validation/mean_surface_sphere": md_dists[1]})
        metrics.append({"validation/dice_similarity_cube": dsc_sims[0]})
        metrics.append({"validation/dice_similarity_sphere": dsc_sims[1]})
        metrics.append({"validation/bending_energy_crude": bending_energy_crude})

    if result.instance.collection == Collection.LEARN:
        jac_det = jacobian_determinant(result.dvf, plot=False)[1]
        sdlogj = (np.log(jac_det) * result.instance.mask).std()
        dsc = dice_similarity_(result.instance.mask, result.deformed_mask, 1.0)
        bending_energy_crude = bending_energy(result.dvf, 3, result.instance.mask)

        metrics.append({"validation/SDLogJ": sdlogj})
        metrics.append({"validation/dice_similarity": dsc})
        metrics.append({"validation/bending_energy_crude": bending_energy_crude})

        logger.info(f"SDLogJ: {sdlogj:.4f}")
        logger.info(f"Dice Similarity: {dsc:.4f}")
        logger.info(f"Bending Energy (crude): {bending_energy_crude:.4f}")

    return metrics


def validation_visualization(result: RunResult, clim_dvf=(None, None), clim_jac=(None, None), tre=True):
    """Calculate all validation visualizations."""
    figs = []  # aggregate all figures
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
        plot_cpoints(result, ax=axes[0, 1])
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

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        plot_color_diff(result, (slice(None), slice(None), 50), ax=axes[0, 0])
        plot_color_diff(result, (slice(None), 50, slice(None)), ax=axes[0, 1])
        plot_color_diff(result, (120, slice(None), slice(None)), ax=axes[0, 2])

        # plot_dvf_masked(result, (slice(None), slice(None), 50), ax=axes[1, 0])
        # plot_dvf_masked(result, (slice(None), 50, slice(None)), ax=axes[1, 1])
        # plot_dvf_masked(result, (120, slice(None), slice(None)), ax=axes[1, 2])

        plot_deformed_mesh(result, (slice(None), slice(None), 50), ax=axes[1, 0], fix_axes=True)
        plot_deformed_mesh(result, (slice(None), 50, slice(None)), ax=axes[1, 1], fix_axes=True)
        plot_deformed_mesh(result, (120, slice(None), slice(None)), ax=axes[1, 2], fix_axes=True)

        jacobian_determinant_masked(result, (slice(None), slice(None), 50), ax=axes[2, 0])
        jacobian_determinant_masked(result, (slice(None), 50, slice(None)), ax=axes[2, 1])
        jacobian_determinant_masked(result, (120, slice(None), slice(None)), ax=axes[2, 2])

        axes[0, 0].set_title("Deformed source vs target (side)")
        axes[0, 1].set_title("Deformed source vs target (front)")
        axes[0, 2].set_title("Deformed source vs target (top)")
        axes[1, 0].set_title("DVF (side)")
        axes[1, 1].set_title("DVF (front)")
        axes[1, 2].set_title("DVF (top)")
        axes[2, 0].set_title("Jacobian determinant (side)")
        axes[2, 1].set_title("Jacobian determinant (front)")
        axes[2, 2].set_title("Jacobian determinant (top)")
        fig.tight_layout()

        figs.append(to_dict(wandb.Image(fig), "overview"))
        figs.append(to_dict(wandb.Image(plot_dvf_3d(result)), "dvf_3d"))

    if tre:
        tre_fig = tre_hist(result.deformed_lms, instance.lms_moving, instance.spacing)
        figs.append(to_dict(wandb.Image(tre_fig), "tre_hist"))

    return figs


def get_vmin_vmax(result1: RunResult, result2: RunResult):
    result1_mag = np.sqrt(
        result1.dvf[result1.dvf.shape[0] // 2, :, :, 0] ** 2 + result1.dvf[result1.dvf.shape[0] // 2, :, :, 1] ** 2
    )
    result2_mag = np.sqrt(
        result2.dvf[result2.dvf.shape[0] // 2, :, :, 0] ** 2 + result2.dvf[result2.dvf.shape[0] // 2, :, :, 1] ** 2
    )
    vmin_dvf = np.min([np.min(result1_mag), np.min(result2_mag)])
    vmax_dvf = np.max([np.max(result1_mag), np.max(result2_mag)])

    jac_hybrid = jacobian_determinant(result1.dvf, plot=False)[0]
    jac_baseline = jacobian_determinant(result2.dvf, plot=False)[0]
    vmin_jac = np.min([np.min(jac_baseline), np.min(jac_hybrid)])
    vmax_jac = np.max([np.max(jac_baseline), np.max(jac_hybrid)])

    return (vmin_dvf, vmax_dvf), (vmin_jac, vmax_jac)


def get_indices_xy(slice_tuple, inv=True):
    indices_xy = np.where(np.array(slice_tuple) == slice(None, None, None))[0]
    if inv:
        indices_xy = list(map(lambda x: INV_MAPPING[x], np.flip(indices_xy)))
    return indices_xy


def mesh_grids(origin, spacing, indices, size, dims, indexing="xy"):
    return np.meshgrid(
        *[
            np.arange(
                origin[indices[i]],
                origin[indices[i]] + spacing[indices[i]] * size[indices[i]],
                spacing[indices[i]],
            )
            for i in range(dims)
        ],
        indexing=indexing,
    )
