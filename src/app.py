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
python app.py

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
        self.on_scroll = None   # callback: (step, event)

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
        self.affine = affine
        self.is_rgb = (self.volume.ndim == 4 and self.volume.shape[-1] == 3)
         # Make volume shape consistent: (X,Y,Z,[3])
        self.shape = self.volume.shape[:3]  # (X, Y, Z)

        # initial crosshair at center
        self.pos = [s // 2 for s in self.shape]  # initial crosshair at center

        # ----------------------- TOP TOOL BAR -----------------------
        toolbar = QtWidgets.QToolBar("MainToolbar")
        self.addToolBar(toolbar)
        load_btn = QtWidgets.QAction("Load NIfTI...", self)
        load_btn.triggered.connect(self._load_new_volume)
        toolbar.addAction(load_btn)

        # main widget and layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        grid = QtWidgets.QGridLayout(central)

        # ---------- Sliders for each orientation ----------
        self.axial_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.axial_slider.setRange(0, self.shape[2] - 1)
        self.axial_slider.setValue(self.pos[2])
        self.axial_slider.valueChanged.connect(self._slider_axial)

        self.coronal_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.coronal_slider.setRange(0, self.shape[1] - 1)
        self.coronal_slider.setValue(self.pos[1])
        self.coronal_slider.valueChanged.connect(self._slider_coronal)

        self.sagittal_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sagittal_slider.setRange(0, self.shape[0] - 1)
        self.sagittal_slider.setValue(self.pos[0])
        self.sagittal_slider.valueChanged.connect(self._slider_sagittal)

        # Matplotlib canvases
        self.axial_canvas = SliceCanvas(self, title="axial",)
        self.coronal_canvas = SliceCanvas(self, title="coronal",)
        self.sagittal_canvas = SliceCanvas(self, title="saggital",)

        for c in (self.axial_canvas, self.coronal_canvas, self.sagittal_canvas):
            c.enable_interaction()  

        self.axial_canvas.on_drag = self._on_axial_drag
        self.coronal_canvas.on_drag = self._on_coronal_drag
        self.sagittal_canvas.on_drag = self._on_sagittal_drag

        self.axial_canvas.on_scroll = self._on_axial_scroll
        self.coronal_canvas.on_scroll = self._on_coronal_scroll
        self.sagittal_canvas.on_scroll = self._on_sagittal_scroll

        # ----------- layout: slider above each viewer -----------
        grid.addWidget(self.axial_slider,   0, 0)
        grid.addWidget(self.coronal_slider, 0, 1)
        grid.addWidget(self.axial_canvas,   1, 0)
        grid.addWidget(self.coronal_canvas, 1, 1)

        grid.addWidget(self.sagittal_slider, 2, 0)
        grid.addWidget(self.sagittal_canvas, 3, 0)

        # PyVista 3D widget
        self.pv_widget = QtInteractor(self)
        grid.addWidget(self.pv_widget.interactor, 3, 1)

        # Opacity slider for 3D volume
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity_slider.setRange(1, 100)   # 1% to 100%
        self.opacity_slider.setValue(60)       # default opacity
        self.opacity_slider.valueChanged.connect(self._on_opacity_change)
        grid.addWidget(self.opacity_slider, 4, 1)

        # status bar text
        self.status = self.statusBar()
        self._update_status()

        # initialize images and 3D
        self._update_all()
        self._init_3d()

    def _update_status(self):
        self.status.showMessage(f'pos (x,y,z): {self.pos[0]}, {self.pos[1]}, {self.pos[2]}')

    # ---------------- 2D slices ----------------
    def _get_axial(self) -> np.ndarray:
        slice2d = self.volume[:, :, self.pos[2], :] if self.is_rgb else self.volume[:, :, self.pos[2]]
        return np.fliplr(slice2d.T)  # transpose for correct orientation

    def _get_coronal(self) -> np.ndarray:
        slice2d = self.volume[:, self.pos[1], :, :] if self.is_rgb else self.volume[:, self.pos[1], :]
        return np.fliplr(slice2d.T)  # transpose for correct orientation

    def _get_sagittal(self) -> np.ndarray:
        slice2d = self.volume[self.pos[0], :, :, :] if self.is_rgb else self.volume[self.pos[0], :, :]
        return np.fliplr(slice2d.T)  # transpose for correct orientation

    def _update_all(self):
        # update 2D images
        self.axial_canvas.show_slice(self._get_axial())
        self.coronal_canvas.show_slice(self._get_coronal())
        self.sagittal_canvas.show_slice(self._get_sagittal())

        # update crosshairs: compute pixel coords for each canvas
        self.axial_canvas.set_crosshair(self.shape[0] - self.pos[0], self.pos[1])
        self.coronal_canvas.set_crosshair(self.shape[0] - self.pos[0], self.pos[2])
        self.sagittal_canvas.set_crosshair(self.shape[1] - self.pos[1], self.pos[2])

        # update 3D marker
        self._update_3d_marker()
        self._update_status()

        # sync sliders with pos[]
        self.axial_slider.blockSignals(True)
        self.coronal_slider.blockSignals(True)
        self.sagittal_slider.blockSignals(True)

        self.axial_slider.setValue(self.pos[2])
        self.coronal_slider.setValue(self.pos[1])
        self.sagittal_slider.setValue(self.pos[0])

        self.axial_slider.blockSignals(False)
        self.coronal_slider.blockSignals(False)
        self.sagittal_slider.blockSignals(False)

    # ---------------- Drag callbacks ----------------
    def _on_axial_drag(self, xpix, ypix, event):
        if xpix is None or ypix is None:
            return
        self.pos[0] = np.clip(int(self.shape[0] - round(xpix)), 0, self.shape[0]-1)
        self.pos[1] = np.clip(int(round(ypix)), 0, self.shape[1]-1)
        self._update_all()

    def _on_coronal_drag(self, xpix, ypix, event):
        if xpix is None or ypix is None:
            return
        self.pos[0] = np.clip(int(self.shape[0] - round(xpix)), 0, self.shape[0]-1)
        self.pos[2] = np.clip(int(round(ypix)), 0, self.shape[2]-1)
        self._update_all()

    def _on_sagittal_drag(self, xpix, ypix, event):
        if xpix is None or ypix is None:
            return
        self.pos[1] = np.clip(int(self.shape[1] - round(xpix)), 0, self.shape[1]-1)
        self.pos[2] = np.clip(int(round(ypix)), 0, self.shape[2]-1)
        self._update_all()

    # ---------------- Scroll callbacks ----------------
    def _on_axial_scroll(self, step, event):
        # axial = Z axis slice
        self.pos[2] = np.clip(self.pos[2] + step, 0, self.shape[2]-1)
        self._update_all()

    def _on_coronal_scroll(self, step, event):
        # coronal = Y axis slice
        self.pos[1] = np.clip(self.pos[1] + step, 0, self.shape[1]-1)
        self._update_all()

    def _on_sagittal_scroll(self, step, event):
        # sagittal = X axis slice
        self.pos[0] = np.clip(self.pos[0] + step, 0, self.shape[0]-1)
        self._update_all()

    # ---------------- Slider callbacks ----------------
    def _slider_axial(self, value):
        self.pos[2] = value
        self._update_all()

    def _slider_coronal(self, value):
        self.pos[1] = value
        self._update_all()

    def _slider_sagittal(self, value):
        self.pos[0] = value
        self._update_all()

    # ---------------- Load new volume ----------------
    def _load_new_volume(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load NIfTI File", "", "NIfTI (*.nii *.nii.gz)"
        )
        if not path:
            return

        data, aff = load_volume(path)
        self.volume = data.astype(float)
        self.affine = aff
        self.shape = self.volume.shape[:3]
        self.pos = [s // 2 for s in self.shape]

        # update slider ranges
        self.axial_slider.setRange(0, self.shape[2] - 1)
        self.coronal_slider.setRange(0, self.shape[1] - 1)
        self.sagittal_slider.setRange(0, self.shape[0] - 1)

        self._update_all()

    # ----------------- PyVista 3D -----------------
    def _init_3d(self):
        pass

    def _update_3d_marker(self):
        pass

    def _on_opacity_change(self, value):
        pass


# Utility loader
def load_volume(path):
    base, ext = os.path.splitext(path)
    if ext == '.gz' and base.endswith('.nii'):
        ext = '.nii.gz'
    ext = ext.lower()
    if ext in ['.nii', '.nii.gz']:
        nii = nib.load(path)
        data = nii.get_fdata(dtype=np.float32)
        aff = nii.affine
        return data, aff
    else:
        raise ValueError('Unsupported extension: ' + ext)

def main():
    vol, aff = load_volume("./subject_001_T1_native_restored.nii.gz")
    print(f"{aff=}")
    print(f"{vol.shape=}")

    app = QtWidgets.QApplication(sys.argv)
    viewer = ViewerApp(vol, affine=aff)
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
