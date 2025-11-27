"""
Interactive 3D + orthogonal 2D viewer (PyQt5 + matplotlib + pyvista)
Features:
- Load NIfTI (.nii/.nii.gz) or .npy numpy arrays
- Shows Axial, Coronal, Sagittal 2D views (matplotlib) with red crosshairs
- Interactive: click & drag in any 2D view to move the crosshair and update all views
- 3D rendering (pyvista) with a red sphere showing the current crosshair position

Dependencies:
- numpy
- nibabel
- PyQt5
- matplotlib
- pyvista
- pyvistaqt

Install (recommended in a venv):
pip install numpy nibabel PyQt5 matplotlib pyvista pyvistaqt

Run:
python 3D_and_Orthogonal_Viewer.py /path/to/volume.nii
Or load a .npy file. If no path provided, a small demo volume will be generated.

Notes:
- This is a single-file example intended to be a practical starting point. For very large volumes
  you might want to use downsampling or streaming volume rendering.
- Orientation assumptions: data is treated as (Z, Y, X) (i.e., axial is along axis 0). If your
  NIfTI uses a different orientation you may need to reorder axes after loading.
"""

import sys
import os
import numpy as np
import nibabel as nib
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5 import QtCore
import vtk


class SliceCanvas(FigureCanvas):
    """A matplotlib canvas showing a single 2D slice with a red crosshair."""
    def __init__(self, parent=None, title="", figsize=(32,32)):
        fig = Figure(figsize=figsize)
        super().__init__(fig)
        self.setParent(parent)
        self.ax = fig.add_subplot(111)
        self.ax.set_title(title)
        self.im = None
        self.vline = None
        self.hline = None
        self.cid_press = None
        self.cid_move = None
        self.cid_scroll = None
        self.pressed = False
        self.on_drag = None  # callback (xpix, ypix, event) -> None
        self.cid_release = None
        self.on_scroll = None   # new callback: (step, event)

    def show_slice(self, slice2d):
        """slice2d: HxW (grayscale) or HxWx3 (RGB)"""
        if slice2d.ndim == 3 and slice2d.shape[2] == 3:
            slice2d = slice2d.astype(float)
            if slice2d.max() > 1.0:
                slice2d /= 255.0
            cmap = None
        else:
            cmap = 'gray'

        if self.im is None:
            self.im = self.ax.imshow(slice2d, cmap=cmap, origin='lower', interpolation='nearest')
            self.ax.axis('off')
        else:
            self.im.set_data(slice2d)
            if cmap:
                self.im.set_clim(np.nanmin(slice2d), np.nanmax(slice2d))
        self.draw_idle()

    def set_crosshair(self, x, y):
        # x,y are in pixel coordinates (cols, rows)
        if self.vline is None:
            self.vline = self.ax.axvline(x, color='r')
            self.hline = self.ax.axhline(y, color='r')
        else:
            self.vline.set_xdata([x,x])
            self.hline.set_ydata([y,y])
        self.draw_idle()

    def enable_interaction(self):
        self.cid_press = self.mpl_connect('button_press_event', self._on_press)
        self.cid_move = self.mpl_connect('motion_notify_event', self._on_move)
        self.cid_release = self.mpl_connect('button_release_event', self._on_release)
        self.cid_scroll = self.mpl_connect('scroll_event', self._on_scroll)

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.pressed = True
        if self.on_drag:
            self.on_drag(event.xdata, event.ydata, event)

    def _on_move(self, event):
        if not self.pressed:
            return
        if event.inaxes != self.ax:
            return
        if self.on_drag:
            self.on_drag(event.xdata, event.ydata, event)

    def _on_release(self, event):
        self.pressed = False

    def disable_interaction(self):
        if self.cid_press: self.mpl_disconnect(self.cid_press)
        if self.cid_move: self.mpl_disconnect(self.cid_move)
        self.pressed = False

    def _on_scroll(self, event):
        if self.on_scroll:
            step = 1 if event.button == 'up' else -1
            self.on_scroll(step, event)

class ViewerApp(QtWidgets.QMainWindow):
    def __init__(self, volume, affine=None):
        super().__init__()
        self.setWindowTitle('3D + Orthogonal Viewer')
        self.volume = volume.astype(float)
        self.volume = np.rot90(self.volume, k=-1, axes=(1, 2))
        self.volume = np.rot90(self.volume, k=-1, axes=(0, 1))
        self.volume = np.rot90(self.volume, k=1, axes=(1, 2))
        self.affine = affine
        self.is_rgb = (self.volume.ndim == 4 and self.volume.shape[-1] == 3)
         # Make volume shape consistent: (X,Y,Z,[3])
        self.shape = self.volume.shape[:3]  # (X, Y, Z)

        # initial crosshair at center
        self.pos = [s // 2 for s in self.shape]  # initial crosshair at center

        # main widget and layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        grid = QtWidgets.QGridLayout(central)

        # Matplotlib canvases
        self.axial_canvas = SliceCanvas(self, title="1",)
        self.coronal_canvas = SliceCanvas(self, title="2",)
        self.sagittal_canvas = SliceCanvas(self, title="3",)

        self.axial_canvas.enable_interaction()
        self.coronal_canvas.enable_interaction()
        self.sagittal_canvas.enable_interaction()

        self.axial_canvas.on_drag = self._on_axial_drag
        self.coronal_canvas.on_drag = self._on_coronal_drag
        self.sagittal_canvas.on_drag = self._on_sagittal_drag

        self.axial_canvas.on_scroll = self._on_axial_scroll
        self.coronal_canvas.on_scroll = self._on_coronal_scroll
        self.sagittal_canvas.on_scroll = self._on_sagittal_scroll

        # PyVista 3D widget
        self.pv_widget = QtInteractor(self)

        # Opacity slider for 3D volume
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity_slider.setRange(1, 100)   # 1% to 100%
        self.opacity_slider.setValue(60)       # default opacity
        self.opacity_slider.valueChanged.connect(self._on_opacity_change)

        # place widgets in grid
        grid.addWidget(self.axial_canvas, 0, 0)
        grid.addWidget(self.coronal_canvas, 0, 1)
        grid.addWidget(self.sagittal_canvas, 1, 0)
        grid.addWidget(self.pv_widget.interactor, 1, 1)
        grid.addWidget(self.opacity_slider, 2, 1)  # slider under the 3D view

        # status bar text
        self.status = self.statusBar()
        self._update_status()

        # initialize images and 3D
        self._update_all()
        # self._init_3d()

    def _update_status(self):
        self.status.showMessage(f'pos (x,y,z): {self.pos[0]}, {self.pos[1]}, {self.pos[2]}')

    # ---------------- 2D slices ----------------
    def _get_axial(self):
        return self.volume[self.pos[0], :, :, :] if self.is_rgb else self.volume[self.pos[0], :, :]

    def _get_coronal(self):
        return self.volume[:, self.pos[1], :, :] if self.is_rgb else self.volume[:, self.pos[1], :]

    def _get_sagittal(self):
        return self.volume[:, :, self.pos[2], :] if self.is_rgb else self.volume[:, :, self.pos[2]]

    def _update_all(self):
        # update 2D images
        self.axial_canvas.show_slice(self._get_axial())
        self.coronal_canvas.show_slice(self._get_coronal())
        self.sagittal_canvas.show_slice(self._get_sagittal())

        # update crosshairs: compute pixel coords for each canvas
        self.axial_canvas.set_crosshair(self.pos[2], self.pos[1])
        self.coronal_canvas.set_crosshair(self.pos[2], self.pos[0])
        self.sagittal_canvas.set_crosshair(self.pos[1], self.pos[0])

        # update 3D marker
        self._update_3d_marker()
        self._update_status()

    # ---------------- Drag callbacks ----------------
    def _on_axial_drag(self, xpix, ypix, event):
        if xpix is None or ypix is None:
            return
        self.pos[1] = np.clip(int(round(ypix)), 0, self.shape[1]-1)
        self.pos[2] = np.clip(int(round(xpix)), 0, self.shape[2]-1)
        self._update_all()

    def _on_coronal_drag(self, xpix, ypix, event):
        if xpix is None or ypix is None:
            return
        self.pos[0] = np.clip(int(round(ypix)), 0, self.shape[0]-1)
        self.pos[2] = np.clip(int(round(xpix)), 0, self.shape[2]-1)
        self._update_all()

    def _on_sagittal_drag(self, xpix, ypix, event):
        if xpix is None or ypix is None:
            return
        self.pos[1] = np.clip(int(round(xpix)), 0, self.shape[1]-1)
        self.pos[0] = np.clip(int(round(ypix)), 0, self.shape[0]-1)
        self._update_all()

    # ---------------- Scroll callbacks ----------------
    def _on_axial_scroll(self, step, event):
        # axial = X axis slice
        self.pos[0] = np.clip(self.pos[0] + step, 0, self.shape[0]-1)
        self._update_all()

    def _on_coronal_scroll(self, step, event):
        # coronal = Y axis slice
        self.pos[1] = np.clip(self.pos[1] + step, 0, self.shape[1]-1)
        self._update_all()

    def _on_sagittal_scroll(self, step, event):
        # sagittal = Z axis slice
        self.pos[2] = np.clip(self.pos[2] + step, 0, self.shape[2]-1)
        self._update_all()

    # ----------------- PyVista 3D -----------------
    def _init_3d(self):
        self.pv_widget.clear()
        vol = self.volume.astype(np.float32)
        if self.is_rgb:
            # RGB: PyVista expects CxHxW flattened; convert later if needed
            vol_max = vol.max()
            if vol_max > 1: vol /= 255.0

        grid = pv.ImageData()

        grid.dimensions = np.array(self.shape[::-1])
        voxel_sizes = (1,1,1)
        if self.affine is not None:
            voxel_sizes = np.sqrt(np.sum(self.affine[:3,:3]**2, axis=0))
        grid.spacing = tuple(voxel_sizes[::-1])
        grid.origin = (0,0,0)

        flat = vol.flatten(order='F')
        grid.point_data.set_scalars(flat, "values")

        self.vol_actor = self.pv_widget.add_volume(grid, cmap='gray' if not self.is_rgb else None,
                                                   opacity='sigmoid_6')
        self.vol_actor.GetProperty().SetInterpolationTypeToLinear() # Set interpolation to linear for smooth rendering

        self.marker = pv.Sphere(radius=max(self.shape) / 80.0,
                                center=self.pos)
        self.marker_actor = self.pv_widget.add_mesh(self.marker, color='red')
        self.pv_widget.reset_camera()
        self.pv_widget.render()

    def _update_3d_marker(self):
        if hasattr(self, 'marker_actor'):
            self.marker_actor.SetPosition(self.pos)
            self.pv_widget.render()

    def _on_opacity_change(self, value):
        alpha = value / 100.0
        if hasattr(self, "vol_actor"):
            prop = self.vol_actor.GetProperty()
            opacity_fn = prop.GetScalarOpacity()
            opacity_fn.RemoveAllPoints()
            opacity_fn.AddPoint(0, 0.0)
            opacity_fn.AddPoint(255, alpha)
            self.pv_widget.render()


# Utility loader
def load_volume(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.nii', '.gz', '.nii.gz']:
        img = nib.load(path)
        data = img.get_fdata(dtype=np.float32)
        aff = img.affine
        return data, aff
    elif ext == '.npy':
        data = np.load(path)
        return data, None
    else:
        raise ValueError('Unsupported extension: ' + ext)


def main(argv):
    if len(argv) > 1:
        path = argv[1]
        vol, aff = load_volume(path)
    else:
        # demo: create a synthetic 3D gaussian
        print('No path provided — creating demo volume')
        x = np.linspace(-1,1,128)
        X,Y,Z = np.meshgrid(x,x,x, indexing='xy')
        vol = np.exp(-(X**2+Y**2+Z**2)*8)
        aff = None

    # Ensure shape is (Z,Y,X). If user provides (X,Y,Z) try to detect and transpose
    # if vol.ndim != 3 or vol.ndim != 4:
    #     raise ValueError('Input volume must be 3D or 4D')
    # Heuristic: if first dim is small (<10) try to reorder. But for now assume (Z,Y,X)

    app = QtWidgets.QApplication(sys.argv)
    viewer = ViewerApp(vol, affine=aff)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main(sys.argv)

# python app.py .\input_mni305_registered.nii.gz
# python app.py .\evaluation_result_01_colored.nii.gz
# python app.py .\evaluation_result_01_colored_sepecific_labels.nii.gz
