from typing import Tuple
import os
import numpy as np
import nibabel as nib
from nibabel.orientations import aff2axcodes, io_orientation

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
        print('shape:', nii.shape)
        print('affine:\n', nii.affine)
        print('axcodes:', aff2axcodes(nii.affine))
        print('io_orientation:', io_orientation(nii.affine))  # shows axis mapping and signs
        print("zooms:", nii.header.get_zooms())
        print("voxel sizes:", [np.linalg.norm(aff[:3,i]) for i in range(3)])
        sx = np.linalg.norm(aff[:3,0])   # spacing along i axis (rows of data)
        sy = np.linalg.norm(aff[:3,1])   # spacing along j axis (cols of data)
        sz = np.linalg.norm(aff[:3,2])   # spacing along k axis (slice thickness)
        print(f'spacing: sx={sx:.3f}, sy={sy:.3f}, sz={sz:.3f}')
        print("-" * 50)
        return data, aff
    else:
        raise ValueError('Unsupported extension: ' + ext)
    
if __name__ == "__main__":
    data, aff = load_volume("./test_nifti_files/001/subject_001_T1_native_high_res.nii.gz")