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
        print("-" * 50)
        return data, aff
    else:
        raise ValueError('Unsupported extension: ' + ext)