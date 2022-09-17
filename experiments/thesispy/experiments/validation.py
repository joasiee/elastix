from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.spatial import distance
from skimage.filters import threshold_multiotsu
import numpy as np
from thesispy.experiments.instance import Instance
from thesispy.definitions import N_CORES

class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
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
  output = np.matrix(np.zeros(n*n))
  output = output.reshape(n,n)
  max_indices = np.array(dvf_slice.shape) - 1
  ei = np.zeros(n, dtype=int)
  ej = np.zeros(n, dtype=int)
  
  for i in range(n):
    for j in range(i+1):
      ei[i] = 1
      ej[j] = 1
      f1 = dvf_slice[tuple((np.clip(p + ei + ej, 0, max_indices)))]
      f2 = dvf_slice[tuple((np.clip(p + ei - ej, 0, max_indices)))]
      f3 = dvf_slice[tuple((np.clip(p - ei + ej, 0, max_indices)))]
      f4 = dvf_slice[tuple((np.clip(p - ei - ej, 0, max_indices)))]
      numdiff = (f1-f2-f3+f4)/4     
      output[i,j] = numdiff
      output[j,i] = numdiff
      ei[i] = 0
      ej[j] = 0
  return output

def bending_energy_point(dvf, p):
  sum = 0.0
  for dim in range(len(dvf.shape)-1):
    sum += np.square(np.linalg.norm(hessian(dvf[..., dim], p)))
  return sum

def bending_energy(dvf):
  results = ProgressParallel(n_jobs=N_CORES, backend='multiprocessing', total=np.prod(dvf.shape[:-1]))(delayed(bending_energy_point)(dvf, p) for p in np.ndindex(dvf.shape[:-1]))

  return np.sum(results) / np.prod(dvf.shape[:-1])

def dice_similarity(moving_deformed, fixed, levels):
  thresholds_moving = threshold_multiotsu(moving_deformed, classes=levels)
  thresholds_fixed = threshold_multiotsu(fixed, classes=levels)
  regions_moving_deformed = np.digitize(moving_deformed, bins=thresholds_moving)
  regions_fixed = np.digitize(fixed, bins=thresholds_fixed)

  intersection = np.sum((regions_moving_deformed == regions_fixed) & (regions_fixed > 0))
  union = np.sum(regions_moving_deformed > 0 ) + np.sum(regions_fixed > 0)
  return 2.0 * intersection / union

def hausdorff_distance(lms1, lms2, spacing=1):
    return np.max(distance.cdist(lms1, lms2).min(axis=1))

def dvf_rmse(dvf1, dvf2, spacing=1):
    return np.linalg.norm((dvf1 - dvf2) * spacing, axis=3).mean()

def mean_surface_distance(lms1, lms2, spacing=1):
    return np.mean(distance.cdist(lms1, lms2).min(axis=1) * spacing)

def tre(lms1, lms2, spacing=1):
    return np.linalg.norm((lms1 - lms2) * spacing, axis=1).mean()
  
def calc_validation(instance: Instance, deformed, dvf, deformed_lms):
    metrics = {}
    if dvf is not None:
        metrics["bending_energy"] = bending_energy(dvf)
        if instance.dvf is not None:
            metrics["dvf_rmse"] = dvf_rmse(dvf, instance.dvf)
    if deformed is not None:
        metrics["dice_similarity"] = dice_similarity(deformed, instance.fixed, 3)
    if deformed_lms is not None:
        metrics["hausdorff_distance"] = hausdorff_distance(deformed_lms, instance.lms_moving)
        metrics["mean_surface_distance"] = mean_surface_distance(deformed_lms, instance.lms_moving)
        metrics["tre"] = tre(deformed_lms, instance.lms_moving)
    return metrics