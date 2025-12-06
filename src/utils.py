from typing import Tuple
import os
import numpy as np
import nibabel as nib


def load_volume(path, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    base, ext = os.path.splitext(path)
    if ext == '.gz' and base.endswith('.nii'):
        ext = '.nii.gz'
    ext = ext.lower()
    if ext in ['.nii', '.nii.gz']:
        nii = nib.load(path)
        data = nii.get_fdata(dtype=dtype)
        aff = nii.affine
        print(f"{aff=}")
        print(f"{data.shape=}")
        return data, aff
    else:
        raise ValueError('Unsupported extension: ' + ext)