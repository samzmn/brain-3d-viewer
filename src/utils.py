import sys
import os
from pathlib import Path
from typing import Tuple
import numpy as np
import nibabel as nib
from nibabel.orientations import aff2axcodes, io_orientation
from scipy.ndimage import zoom


def resource_path(relative):
    if hasattr(sys, "_MEIPASS"):
        # Running inside PyInstaller bundle
        return Path(sys._MEIPASS) / relative
    else:
        # Running in development
        return Path(__file__).parent / relative

def load_volume(path, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    base, ext = os.path.splitext(path)
    if ext == '.gz' and base.endswith('.nii'):
        ext = '.nii.gz'
    ext = ext.lower()
    if ext in ['.nii', '.nii.gz']:
        nii = nib.load(path)
        nii = nib.as_closest_canonical(nii) # canonical ordering (nibabel’s “RAS” / nearest canonical)
        data = nii.get_fdata(dtype=dtype)
        aff = nii.affine
        # print('path: ', path)
        # print('shape:', nii.shape)
        # print('affine:\n', nii.affine)
        # print('axcodes:', aff2axcodes(nii.affine))
        # print('io_orientation:', io_orientation(nii.affine))  # shows axis mapping and signs
        # print("zooms:", nii.header.get_zooms())
        # print("voxel sizes:", [np.linalg.norm(aff[:3,i]) for i in range(3)])
        # sx = np.linalg.norm(aff[:3,0])   # spacing along i axis (rows of data)
        # sy = np.linalg.norm(aff[:3,1])   # spacing along j axis (cols of data)
        # sz = np.linalg.norm(aff[:3,2])   # spacing along k axis (slice thickness)
        # print(f'spacing: sx={sx:.3f}, sy={sy:.3f}, sz={sz:.3f}')
        # print("-" * 50)
        return data, aff
    else:
        raise ValueError('Unsupported extension: ' + ext)
    
def label_structure(prob_map: np.ndarray, index: int, threshold:float=0.3) -> np.ndarray:
    label_map_np = np.zeros_like(prob_map, dtype=np.uint8)
    mask = (prob_map > threshold).astype(np.uint8)
    label_map_np[mask > 0] = index
    return label_map_np

def upsample_slice(slice2d, scale=2.0, order=1):
    """
    scale: 2.0 → 2x resolution in each dimension
    order: interpolation order
           0 = nearest (for Label maps and masks)
           1 = bilinear (for t1 and t2 MRI)
           3 = bicubic
    """
    return zoom(slice2d, zoom=scale, order=order)

def world_coords_of_axial_slice2d(volume: np.ndarray, slice_idx: int, affine: np.ndarray):
    # axial example: compute world coords for image corners of slice k
    nx, ny, nz = volume.shape
    # corners in voxel index space (i,j,k)
    c00 = affine @ np.array([0, 0, slice_idx, 1])   # world coord of (0,0,k)
    c10 = affine @ np.array([nx, 0, slice_idx, 1])  # (nx,0,k)
    c01 = affine @ np.array([0, ny, slice_idx, 1])  # (0,ny,k)
    c11 = affine @ np.array([nx, ny, slice_idx, 1])
    # choose x axis = j (cols), y axis = i (rows)
    # x runs from c00_x to c01_x, y runs from c00_y to c10_y (be careful with orientation)
    # xmin = c00[0]; xmax = c01[0]
    # ymin = c00[1]; ymax = c10[1]
    print(c00, c10, c01, c11)
    xmin = np.min([c00[0], c01[0]])
    xmax = np.max([c10[0], c11[0]])
    ymin = np.min([c00[1], c10[1]])
    ymax = np.max([c01[1], c11[1]])
    extend = [xmin, xmax, ymin, ymax]
    extend = [ymin, ymax, xmin, xmax]
    print(extend)

    return extend

def world_coords_of_coronal_slice2d(volume: np.ndarray, slice_idx: int, affine: np.ndarray):
    # coronal example: compute world coords for image corners of slice k
    nx, ny, nz = volume.shape
    # corners in voxel index space (i,j,k)
    c00 = affine @ np.array([0, slice_idx, 0, 1])
    c10 = affine @ np.array([nx, slice_idx, 0, 1])
    c01 = affine @ np.array([0, slice_idx, nz, 1])

    # choose x axis = j (cols), y axis = i (rows)
    # x runs from c00_x to c01_x, y runs from c00_y to c10_y (be careful with orientation)
    xmin = c00[0]; xmax = c01[0]
    ymin = c00[1]; ymax = c10[1]

    return [xmin, xmax, ymin, ymax]

def world_coords_of_sagittal_slice2d(volume: np.ndarray, slice_idx: int, affine: np.ndarray):
    # coronal example: compute world coords for image corners of slice k
    nx, ny, nz = volume.shape
    # corners in voxel index space (i,j,k)
    c00 = affine @ np.array([slice_idx, 0, 0, 1])
    c10 = affine @ np.array([slice_idx, ny, 0, 1])
    c01 = affine @ np.array([slice_idx, 0, nz, 1])

    # choose x axis = j (cols), y axis = i (rows)
    # x runs from c00_x to c01_x, y runs from c00_y to c10_y (be careful with orientation)
    xmin = c00[0]; xmax = c01[0]
    ymin = c00[1]; ymax = c10[1]

    return [xmin, xmax, ymin, ymax]

if __name__ == "__main__":
    data, aff = load_volume("./test_nifti_files/001/subject_001_T1_native_high_res.nii.gz")
