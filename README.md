**Brain 3D Viewer**

- **Project:** `brain-3d-viewer`
- **Location:** `src/main.py`
- **Purpose:** Advanced interactive 3D brain MRI viewer for volumetric (NIfTI) data showing orthogonal 2D slices with segmentation tools, supporting T1/T2 modalities, image enhancement filters, and 3D label manipulation for medical imaging analysis.

**Quick Start**

- Create and activate a Python virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Run the viewer from the repository root:

```powershell
python src\main.py
```

If you prefer module style (from repository root):

```powershell
python -m src.main
```

**What the viewer shows**

- Three orthogonal 2D views are provided (matplotlib canvases) plus a 3D rendering:
  - **Axial (top-to-bottom):** slices taken along the Z axis. In the UI this is the `axial` canvas.
  - **Coronal (front-to-back):** slices taken along the Y axis. In the UI this is the `coronal` canvas.
  - **Sagittal (side-to-side):** slices taken along the X axis. In the UI this is the `sagittal` canvas.

**How axes are used in this code**

- The viewer expects a 3D array with indexing `(X, Y, Z)` where:
  - `X` = left-right (sagittal axis)
  - `Y` = anterior-posterior (coronal axis)
  - `Z` = inferior-superior (axial axis)

- Implementation details (see `src/main.py`):
  - Axial slice is extracted as `volume[:, :, z]` and then transformed for display.
  - Coronal slice is `volume[:, y, :]`.
  - Sagittal slice is `volume[x, :, :]`.

**Operations and Features**

- **Loading Data:** Load T1 NIfTI, T2 NIfTI, and segmentation NIfTI files.
- **Modality Switching:** Toggle between T1 and T2 modalities.
- **Segmentation Overlay:** Load and display segmentation with adjustable opacity. Labels can be selected from a panel showing colors and names.
- **Image Enhancement:** Apply filters including denoising, contrast enhancement (CLAHE), and window tightening, restricted to ROI masks.
- **Interactive Navigation:** Click and drag on 2D canvases to move the crosshair; scroll to change slices; use sliders for precise slice selection.
- **Zoom and Pan:** Hold Ctrl while dragging to zoom and pan individual views.
- **View Maximization:** Maximize individual views for detailed inspection.
- **Label Manipulation:** Select a label and use keyboard shortcuts (arrow keys for movement, +/- for resizing, ,/. for rotation) to edit segmentation in 3D space.
- **Save/Load Segmentation:** Save edited segmentations and reload them.

**Controls / Interaction**

- Click and drag on any 2D canvas to move the red crosshair; other views update immediately.
- Use the horizontal sliders beneath each view to pick a slice for that plane.
- Use the mouse scroll wheel over a canvas to step through slices for that plane (scroll up increases slice index).
- Click the small `⤢` button next to a slider to maximize a single view.
- `Zoom Mode`: hold `Ctrl` while dragging to pan/zoom the selected canvas.
- `Show Segmentation` checkbox toggles segmented overlay (if loaded). Opacity slider controls overlay alpha.
- Select labels from the left panel; use keyboard for 3D manipulation when a label is selected and a canvas is focused.

**Data Organization and Naming Standards**

To use the application with its full potential (segmentation, label manipulation, and image enhancement), organize your data in a directory following this structure and naming convention:

- **Main Files:**
  - `subject_XXX_T1_*.nii.gz`: T1-weighted NIfTI file (e.g., `subject_001_T1_pre.nii.gz`)
  - `subject_XXX_T2_*.nii.gz`: T2-weighted NIfTI file (e.g., `subject_001_T2_merged.nii.gz`)
  - `subject_XXX_structures_labeled.nii.gz`: Segmentation NIfTI with labeled structures
  - `subject_XXX_labels.json`: JSON file mapping structure names to indices (e.g., `{"STN_lh": 1, "STN_rh": 2, ...}`)

- **Subdirectories:**
  - `labels/`: Contains probability maps for each structure, named as `{structure_name}_prob_in_subject_XXX.nii.gz` (e.g., `STN_lh_prob_in_subject_001.nii.gz`)
  - `masks/`: Contains mask files, such as `brain_mask_of_subject_XXX.nii.gz` and `second_subcortical_mask_of_subject_XXX.nii.gz` for ROI-based enhancements

Replace `XXX` with the subject identifier (e.g., `001`). This structure enables loading of probability-based structures for advanced label resizing and manipulation.

**Reorienting data (if slices look rotated/flipped or axes appear swapped)**

Detects NIfTI affine and auto-canonicalize the volume on load.

**Code pointers (useful locations in the repo)**

- Viewer main file: `src/main.py`
  - Slice creation functions: `_get_axial()`, `_get_coronal()`, `_get_sagittal()`
  - Crosshair / sync: `_update_all()`
  - Mouse scroll handlers: `_on_axial_scroll()`, `_on_coronal_scroll()`, `_on_sagittal_scroll()`

**Troubleshooting & tips**

- If images appear rotated/flipped: try `nib.as_closest_canonical` before passing data to the viewer.
- If the UI fails to start, verify the `requirements.txt` versions or try a more recent `PyQt5`/`matplotlib` combination.
- For very large volumes consider downsampling before visualization to keep UI responsive.

**Example: run**

```python
# run viewer
python src/main.py
```

**Author**

- **Name:** Sam Zamani
- **Email:** sam.zmn99@gmail.com
- **Github:** https://github.com/samzmn

**License & Credits**

- See `licenses/` for third-party library licenses included with this project.
