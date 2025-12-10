**Brain 3D Viewer**

- **Project:** `brain-3d-viewer`
- **Location:** `src/main.py`
- **Purpose:** Simple interactive viewer for volumetric (NIfTI / NumPy) data showing orthogonal 2D slices

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


**Controls / Interaction**

- Click and drag on any 2D canvas to move the red crosshair; other views update immediately.
- Use the horizontal sliders beneath each view to pick a slice for that plane.
- Use the mouse scroll wheel over a canvas to step through slices for that plane (scroll up increases slice index).
- Click the small `⤢` button next to a slider to maximize a single view.
- `Zoom Mode`: hold `Ctrl` while dragging to pan/zoom the selected canvas.
- `Show Segmentation` checkbox toggles segmented overlay (if loaded). Opacity slider controls overlay alpha.

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
python src\main.py
```

**License & Credits**

- See `licenses/` for third-party library licenses included with this project.
