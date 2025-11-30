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
            self.im = self.ax.imshow(slice2d, cmap=cmap, origin='upper', interpolation='nearest')
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

        # main widget and layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        grid = QtWidgets.QGridLayout(central)

        # Matplotlib canvases
        self.axial_canvas = SliceCanvas(self, title="axial",)
        self.coronal_canvas = SliceCanvas(self, title="coronal",)
        self.sagittal_canvas = SliceCanvas(self, title="sagittal",)

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
        self._init_3d()

    def _update_status(self):
        self.status.showMessage(f'pos (x,y,z): {self.pos[0]}, {self.pos[1]}, {self.pos[2]}')

    # ---------------- 2D slices ----------------
    def _get_sagittal(self):
        return self.volume[self.pos[0], :, :, :] if self.is_rgb else self.volume[self.pos[0], :, :]

    def _get_coronal(self):
        return self.volume[:, self.pos[1], :, :] if self.is_rgb else self.volume[:, self.pos[1], :]

    def _get_axial(self):
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
        pass

    def _update_3d_marker(self):
        pass

    def _on_opacity_change(self, value):
        pass


# Utility loader
def load_volume(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.nii', '.gz', '.nii.gz']:
        nii = nib.load(path)

        # canonicalize to RAS and get the reoriented array
        # nii_canon = nib.as_closest_canonical(nii)
        data = nii.get_fdata(dtype=np.float32)
        aff = nii.affine

        # ONE-TIME transform so displayed slices do NOT require per-slice transpose/flip.
        # This matches the earlier behaviour where you used `.T` + `np.fliplr` for each slice.
        # Transform: transpose axes (0,2,1) then flip the last axis.
        # Resulting shape: (X, Z, Y)
        # data = np.transpose(data, (0, 2, 1))
        # data = np.flip(data, axis=2)
        # data = np.rot90(data, k=1, axes=(1,2))
        # data = np.rot90(data, k=2, axes=(0,1))
        # data = np.rot90(data, k=2, axes=(1,2))
        # data = np.flip(data, axis=1)
        # data = np.rot90(data, k=1, axes=(0,2))
        # data = np.flip(data, axis=0)
        # data = np.flip(data, axis=2)
        # data = np.flip(data, axis=1)
        # data = np.rot90(data, k=1, axes=(0,2))

        return data, aff
    elif ext == '.npy':
        data = np.load(path)
        return data, None
    else:
        raise ValueError('Unsupported extension: ' + ext)

def main():
    vol, aff = load_volume("./subject_001_T1_native_restored.nii.gz")
    print(aff)
    print(vol.shape)

    app = QtWidgets.QApplication(sys.argv)
    viewer = ViewerApp(vol, affine=aff)
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
