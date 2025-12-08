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
from turtle import color
from typing import Tuple
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, rotate
import nibabel as nib
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches
import vtk
from utils import load_volume

# ------------------------------------------------------------
# LabelPanel: shows label colors + selection
# ------------------------------------------------------------
class LabelPanel(QtWidgets.QWidget):
    label_selected = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)

        self.label_list = QtWidgets.QListWidget()
        self.label_list.itemClicked.connect(self.on_item_clicked)

        self.current_label_display = QtWidgets.QLabel("Selected Label: None")

        layout.addWidget(QtWidgets.QLabel("Labels:"))
        layout.addWidget(self.label_list)
        layout.addWidget(self.current_label_display)
        
        self.setMaximumWidth(220)

        self.selected_label = None

    def set_labels(self, seg_colors):
        self.label_list.clear()
        for label, color in seg_colors.items():
            item = QtWidgets.QListWidgetItem(f"Label {label}")
            pix = QtGui.QPixmap(20,20)
            pix.fill(QtGui.QColor(*(int(c*255) for c in color)))
            item.setIcon(QtGui.QIcon(pix))
            item.setData(QtCore.Qt.UserRole, label)
            self.label_list.addItem(item)

    def on_item_clicked(self, item):
        label = item.data(QtCore.Qt.UserRole)
        self.current_label_display.setText(f"Selected Label: {label}")
        self.label_selected.emit(label)
        self.selected_label = label
        
class SliceCanvas(FigureCanvas):
    """A matplotlib canvas showing a single 2D slice with a red crosshair."""
    def __init__(self, parent=None, title="", figsize=(32,32)):
        self.fig = Figure(figsize=figsize)
        self.fig.tight_layout()
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(title)
        self.title = title
        self.shape = None
        self.im = None
        self.seg = None
        self.vline = None
        self.hline = None
        self.cid_press = None
        self.cid_move = None
        self.cid_scroll = None
        self.pressed = False
        self.on_drag = None # callback (xpix, ypix, event) -> None
        self.cid_release = None
        self.on_scroll = None # callback: (step, event)
        # self.cid_key_press = None
        # self.on_key_press = None # callback: (event)

        self.is_focused = False
        self.zoom_enabled = False # controlled by ViewerApp checkbox
        self.zoom_scale = 1.0 # current zoom factor
        self.prev_drag = None  # for panning

        self.focus_rect = patches.Rectangle(
            (0, 0), 1, 1,
            linewidth=4.0,
            edgecolor=(1, 0, 0, 0.9),
            facecolor="none",
            transform=self.ax.transAxes,   # important: match axes coordinates
            zorder=1000,
            visible=False
        )
        self.ax.add_patch(self.focus_rect)
    
    def draw_focus_border(self):
        self.focus_rect.set_visible(self.is_focused)
        self.draw_idle()

    def show_slice(self, slice2d, seg_slice2d=None):
        """slice2d: HxW (grayscale) or HxWx3 (RGB)"""
        if slice2d.ndim == 3 and slice2d.shape[2] == 3:
            slice2d = slice2d.astype(float)
            if slice2d.max() > 1.0:
                slice2d /= 255.0
            cmap = None
        else:
            cmap = 'gray'

        if self.shape is None:
            self.shape = slice2d.shape[:2]

        if self.im is None:
            self.ax.axis('off')
            self.im = self.ax.imshow(slice2d, cmap=cmap, origin='lower', interpolation='nearest')
        else:
            self.im.set_data(slice2d)
            if cmap:
                self.im.set_clim(np.nanmin(slice2d), np.nanmax(slice2d))
        
        if self.seg is None:
            if seg_slice2d is not None:
                self.seg = self.ax.imshow(seg_slice2d, origin='lower', interpolation='nearest')
        else:
            if seg_slice2d is not None:
                self.seg.set_data(seg_slice2d)
            else:
                self.seg.remove()
                self.seg = None

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
        # self.cid_key_press = self.mpl_connect("key_press_event", self._on_key_press)

    def disable_interaction(self):
        if self.cid_press: self.mpl_disconnect(self.cid_press)
        if self.cid_move: self.mpl_disconnect(self.cid_move)
        self.pressed = False

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.pressed = True
        if self.on_drag and not self.zoom_enabled:
            self.on_drag(event.xdata, event.ydata, event)
        self.parent().parent().canvas_clicked(self)  # notify main app
        self.prev_drag = (event.xdata, event.ydata)  # for panning

    def _on_move(self, event):
        if not self.pressed:
            return
        if event.inaxes != self.ax:
            return
        if self.zoom_enabled:
            self._on_pan(event)
        else:
            if self.on_drag:
                self.on_drag(event.xdata, event.ydata, event)

    def _on_release(self, event):
        self.pressed = False
        self.prev_drag = None

    def _on_scroll(self, event):
        step = 1 if event.button == 'up' else -1
        if self.zoom_enabled:
            self._on_zoom(step, event)
        else:
            if self.on_scroll is not None:
                self.on_scroll(step, event) # normal callback to viewer

    def _on_pan(self, event):
        if self.prev_drag is None or event.xdata is None or event.ydata is None:
            return

        dx = event.xdata - self.prev_drag[0]
        dy = event.ydata - self.prev_drag[1]

        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()

        new_x0 = x0 - dx
        new_x1 = x1 - dx
        new_y0 = y0 - dy
        new_y1 = y1 - dy

        # clamp to image boundaries
        h, w = self.shape[:2]
        if new_x0 < 0:
            new_x1 -= new_x0
            new_x0 = 0
        if new_y0 < 0:
            new_y1 -= new_y0
            new_y0 = 0
        if new_x1 > w:
            diff = new_x1 - w
            new_x0 -= diff
            new_x1 = w
        if new_y1 > h:
            diff = new_y1 - h
            new_y0 -= diff
            new_y1 = h

        # apply
        self.ax.set_xlim(new_x0, new_x1)
        self.ax.set_ylim(new_y0, new_y1)
        self.prev_drag = (event.xdata, event.ydata)
        self.draw_idle()

    def _on_zoom(self, step, event):
        if event.xdata is None or event.ydata is None:
            return  # ignore zooming outside the image

        # compute new scale
        factor = 1.1 if step > 0 else 1/1.1
        new_scale = np.clip(self.zoom_scale * factor, 1.0, 20.0)
        factor = new_scale / self.zoom_scale   # true zoom ratio
        self.zoom_scale = new_scale

        cx, cy = event.xdata, event.ydata
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()

        # scale limits around cursor
        new_x0 = cx - (cx - x0) / factor
        new_x1 = cx + (x1 - cx) / factor
        new_y0 = cy - (cy - y0) / factor
        new_y1 = cy + (y1 - cy) / factor

        h, w = self.shape[:2]

        if self.zoom_scale == 1.0:
            # fully zoomed out → reset entire image
            self.ax.set_xlim(0, w)
            self.ax.set_ylim(0, h)
        else:
            # only clamp when zooming out
            if step < 0:  # zoom out
                new_x0 = max(0, new_x0)
                new_y0 = max(0, new_y0)
                new_x1 = min(w, new_x1)
                new_y1 = min(h, new_y1)

            self.ax.set_xlim(new_x0, new_x1)
            self.ax.set_ylim(new_y0, new_y1)

        self.draw_idle()

    # def _on_key_press(self, event):
    #     if self.on_key_press is not None:
    #         self.on_key_press(event)

class ViewerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Brain Viewer with Segmentation Tools')
        self.volume = None
        self.affine = None
        self.is_rgb = None
        self.shape = None  # (X, Y, Z)
        self.pos = None
        self.seg_volume = None
        self.seg_rgba = None
        self.label_colors = {}
        self.focused_canvas = None

        # ----------------------- TOP TOOL BAR -----------------------
        toolbar = QtWidgets.QToolBar("MainToolbar")
        self.addToolBar(toolbar)

        load_btn = QtWidgets.QAction("Load T1 NIfTI...", self)
        load_btn.triggered.connect(self._load_new_volume)
        toolbar.addAction(load_btn)

        load_seg_btn = QtWidgets.QAction("Load Segmentation NIfTI...", self)
        load_seg_btn.triggered.connect(self._load_segmentation_nifti)
        toolbar.addAction(load_seg_btn)

        save_seg_btn = QtWidgets.QAction("Save Segmentation NIfTI...", self)
        save_seg_btn.triggered.connect(self._save_segmentation_nifti)
        toolbar.addAction(save_seg_btn)

        reload_seg_btn = QtWidgets.QAction("Reload Segmentation", self)
        reload_seg_btn.triggered.connect(self._reload_segmentation_nifti)
        toolbar.addAction(reload_seg_btn)

        self.seg_checkbox = QtWidgets.QCheckBox("Show Segmentation")
        self.seg_checkbox.setChecked(True)
        self.seg_checkbox.stateChanged.connect(self._toggle_seg_visibility)
        toolbar.addWidget(self.seg_checkbox)

        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setSingleStep(10)
        self.opacity_slider.setPageStep(10)
        self.opacity_slider.valueChanged.connect(self._change_opacity)
        toolbar.addWidget(QtWidgets.QLabel("Opacity"))
        toolbar.addWidget(self.opacity_slider)

        self.zoom_checkbox = QtWidgets.QCheckBox("Zoom Mode")
        self.zoom_checkbox.setChecked(False)
        self.zoom_checkbox.stateChanged.connect(self._toggle_zoom_mode_checkbox)
        toolbar.addWidget(self.zoom_checkbox)

        # ----------------------- MAIN WIDGET and layout -----------------------
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # ---------------- Left panel (labels) ----------------------
        self.label_panel = LabelPanel()
        layout.addWidget(self.label_panel)

        # ---------------- 2D views container ---------------------
        self.grid_layout = QtWidgets.QGridLayout()
        layout.addLayout(self.grid_layout)

        # ---------- Sliders for each orientation ----------
        self.axial_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.axial_slider.valueChanged.connect(self._slider_axial)

        self.coronal_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.coronal_slider.valueChanged.connect(self._slider_coronal)

        self.sagittal_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sagittal_slider.valueChanged.connect(self._slider_sagittal)

        self.axial_label = QtWidgets.QLabel("0")
        self.coronal_label = QtWidgets.QLabel("0")
        self.sagittal_label = QtWidgets.QLabel("0")

        self.axial_max_btn = QtWidgets.QPushButton("⤢")
        self.coronal_max_btn = QtWidgets.QPushButton("⤢")
        self.sagittal_max_btn = QtWidgets.QPushButton("⤢")

        self.axial_max_btn.clicked.connect(lambda: self._toggle_maximize("axial"))
        self.coronal_max_btn.clicked.connect(lambda: self._toggle_maximize("coronal"))
        self.sagittal_max_btn.clicked.connect(lambda: self._toggle_maximize("sagittal"))

        self.maximized_view = None

        # Matplotlib canvases
        self.axial_canvas = SliceCanvas(self, title="axial",)
        self.coronal_canvas = SliceCanvas(self, title="coronal",)
        self.sagittal_canvas = SliceCanvas(self, title="sagittal",)

        for c in (self.axial_canvas, self.coronal_canvas, self.sagittal_canvas):
            c.enable_interaction()  

        self.axial_canvas.on_drag = self._on_axial_drag
        self.coronal_canvas.on_drag = self._on_coronal_drag
        self.sagittal_canvas.on_drag = self._on_sagittal_drag

        self.axial_canvas.on_scroll = self._on_axial_scroll
        self.coronal_canvas.on_scroll = self._on_coronal_scroll
        self.sagittal_canvas.on_scroll = self._on_sagittal_scroll

        # self.axial_canvas.on_key_press = self._on_key_press
        # self.coronal_canvas.on_key_press = self._on_key_press
        # self.sagittal_canvas.on_key_press = self._on_key_press

        QtWidgets.QApplication.instance().installEventFilter(self)

        # ----------- layout with label + slider + maximize button -----------
        self._rebuild_grid_layout()

        # status bar text
        self.status = self.statusBar()
        self._update_status()

        # initialize images and 3D
        # self._update_all()
        self._test_init()

    def _update_status(self):
        if self.pos is not None:
            self.status.showMessage(f'pos (x,y,z): {self.pos[0]}, {self.pos[1]}, {self.pos[2]}')

    # ---------------- 2D slices ----------------
    def _make_seg_overlay(self, seg2d):
        """Given a seg2d RGBA slice, apply opacity and return overlay."""
        overlay = seg2d.copy()
        overlay[..., 3] *= (self.opacity_slider.value() / 100.0) # Alpha
        overlay = np.transpose(overlay, (1, 0, 2))
        overlay = np.fliplr(overlay)
        return overlay

    def _get_axial(self) -> np.ndarray:
        slice2d = self.volume[:, :, self.pos[2], :] if self.is_rgb else self.volume[:, :, self.pos[2]]
        slice2d = np.fliplr(slice2d.T)  # transpose for correct orientation
        if self.seg_rgba is not None and self.seg_checkbox.isChecked():
            seg_slice2d = self._make_seg_overlay(self.seg_rgba[:, :, self.pos[2]])
        else:
            seg_slice2d = None
        return slice2d, seg_slice2d

    def _get_coronal(self) -> np.ndarray:
        slice2d = self.volume[:, self.pos[1], :, :] if self.is_rgb else self.volume[:, self.pos[1], :]
        slice2d = np.fliplr(slice2d.T)  # transpose for correct orientation
        if self.seg_rgba is not None and self.seg_checkbox.isChecked():
            seg_slice2d = self._make_seg_overlay(self.seg_rgba[:, self.pos[1], :])
        else:
            seg_slice2d = None
        return slice2d, seg_slice2d

    def _get_sagittal(self) -> np.ndarray:
        slice2d = self.volume[self.pos[0], :, :, :] if self.is_rgb else self.volume[self.pos[0], :, :]
        slice2d = np.fliplr(slice2d.T)  # transpose for correct orientation
        if self.seg_rgba is not None and self.seg_checkbox.isChecked():
            seg_slice2d = self._make_seg_overlay(self.seg_rgba[self.pos[0], :, :])
        else:
            seg_slice2d = None
        return slice2d, seg_slice2d

    def _update_all(self):
        # update 2D images
        self.axial_canvas.show_slice(*self._get_axial())
        self.coronal_canvas.show_slice(*self._get_coronal())
        self.sagittal_canvas.show_slice(*self._get_sagittal())

        # update crosshairs: compute pixel coords for each canvas
        self.axial_canvas.set_crosshair(self.shape[0] - self.pos[0], self.pos[1])
        self.coronal_canvas.set_crosshair(self.shape[0] - self.pos[0], self.pos[2])
        self.sagittal_canvas.set_crosshair(self.shape[1] - self.pos[1], self.pos[2])

        # update 3D marker
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

    # ---------------- Canvas click handling ----------------
    def canvas_clicked(self, canvas):
        for c in [self.axial_canvas, self.sagittal_canvas, self.coronal_canvas]:
            c.is_focused = (c is canvas)
            c.draw_focus_border()
        self.focused_canvas = canvas

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
        self.axial_label.setText(str(self.pos[2] + 1))
        self._update_all()

    def _on_coronal_scroll(self, step, event):
        # coronal = Y axis slice
        self.pos[1] = np.clip(self.pos[1] + step, 0, self.shape[1]-1)
        self.coronal_label.setText(str(self.pos[1] + 1))
        self._update_all()

    def _on_sagittal_scroll(self, step, event):
        # sagittal = X axis slice
        self.pos[0] = np.clip(self.pos[0] + step, 0, self.shape[0]-1)
        self.sagittal_label.setText(str(self.pos[0] + 1))
        self._update_all()

    # ---------------- Slider callbacks ----------------
    def _slider_axial(self, value):
        self.axial_label.setText(str(value + 1))
        self.pos[2] = value
        self._update_all()

    def _slider_coronal(self, value):
        self.coronal_label.setText(str(value + 1))
        self.pos[1] = value
        self._update_all()

    def _slider_sagittal(self, value):
        self.sagittal_label.setText(str(value + 1))
        self.pos[0] = value
        self._update_all()

    # ---------------- Canvases visibility ----------------
    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
            else:
                sublayout = item.layout()
                if sublayout is not None:
                    self._clear_layout(sublayout)

    def _rebuild_grid_layout(self):
        grid = self.grid_layout  # store this during __init__

        # 1) Remove everything
        self._clear_layout(grid)

        # 2) Re-create slider + labels + buttons
        axial_row = QtWidgets.QHBoxLayout()
        axial_row.addWidget(self.axial_label)
        axial_row.addWidget(self.axial_slider)
        axial_row.addWidget(self.axial_max_btn)
        grid.addLayout(axial_row, 0, 0)

        coronal_row = QtWidgets.QHBoxLayout()
        coronal_row.addWidget(self.coronal_label)
        coronal_row.addWidget(self.coronal_slider)
        coronal_row.addWidget(self.coronal_max_btn)
        grid.addLayout(coronal_row, 0, 1)

        sagittal_row = QtWidgets.QHBoxLayout()
        sagittal_row.addWidget(self.sagittal_label)
        sagittal_row.addWidget(self.sagittal_slider)
        sagittal_row.addWidget(self.sagittal_max_btn)
        grid.addLayout(sagittal_row, 2, 0)

        # 3) Re-add canvases
        grid.addWidget(self.axial_canvas,    1, 0)
        grid.addWidget(self.coronal_canvas,  1, 1)
        grid.addWidget(self.sagittal_canvas, 3, 0)

        # 4) Restore good stretch factors
        grid.setRowStretch(1, 1)
        grid.setRowStretch(3, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        # 5) Update geometry
        # self.centralWidget().updateGeometry()
        self.updateGeometry()
        self.repaint()

    def _maximize_view(self, view):
        grid = self.grid_layout

        # 1) Clear the entire grid
        self._clear_layout(grid)

        vbox = QtWidgets.QVBoxLayout()

        # 2) Add only the selected canvas
        if view == "axial":
            row = QtWidgets.QHBoxLayout()
            row.addWidget(self.axial_label)
            row.addWidget(self.axial_slider)
            row.addWidget(self.axial_max_btn)
            vbox.addLayout(row)
            vbox.addWidget(self.axial_canvas)
        elif view == "coronal":
            row = QtWidgets.QHBoxLayout()
            row.addWidget(self.coronal_label)
            row.addWidget(self.coronal_slider)
            row.addWidget(self.coronal_max_btn)
            vbox.addLayout(row)
            vbox.addWidget(self.coronal_canvas)
        else:
            row = QtWidgets.QHBoxLayout()
            row.addWidget(self.sagittal_label)
            row.addWidget(self.sagittal_slider)
            row.addWidget(self.sagittal_max_btn)
            vbox.addLayout(row)
            vbox.addWidget(self.sagittal_canvas)

        grid.addLayout(vbox, 0, 0, 4, 2)

        # Make it fill all
        grid.setRowStretch(0, 1)
        grid.setColumnStretch(0, 1)

        self.updateGeometry()

    def _toggle_maximize(self, view):
        # --- RESTORE MODE ---
        if self.maximized_view is not None:
            self.maximized_view = None
            self._rebuild_grid_layout()
            return

        # --- MAXIMIZE MODE ---
        self.maximized_view = view
        self._maximize_view(view)

    # ---------------- Key press handling ----------------
    def eventFilter(self, obj, event):
        # Detect key release → disable zoom
        if event.type() == QtCore.QEvent.KeyRelease:
            # When Ctrl is released → disable zoom
            if not (event.modifiers() & QtCore.Qt.ControlModifier):
                self._toggle_zoom_mode(False)
                
        if event.type() == QtCore.QEvent.KeyPress:
            # --- CTRL-drag zoom activation ---
            if event.modifiers() & QtCore.Qt.ControlModifier:
                self._toggle_zoom_mode(True)
            else:
                self._toggle_zoom_mode(False)
            
            key = event.key()
            
            keymap = {
                QtCore.Qt.Key_Up: "up",
                QtCore.Qt.Key_Down: "down",
                QtCore.Qt.Key_Left: "left",
                QtCore.Qt.Key_Right: "right",
                QtCore.Qt.Key_Plus: "+",
                QtCore.Qt.Key_Equal: "+",
                QtCore.Qt.Key_Minus: "-",
                QtCore.Qt.Key_Comma: "rotate_ccw",
                QtCore.Qt.Key_Period: "rotate_cw",
            }

            if key in keymap:
                self._on_key_press(key = keymap[key])
                return True  # prevent any widget from getting this key

        return super().eventFilter(obj, event)
    
    def _on_key_press(self, key):
        if self.label_panel.selected_label is None:
            return
        if key == "+":
            self._resize_label_3d("+")
        elif key == "-":
            self._resize_label_3d("-")
        if self.focused_canvas is None:
            return
        if key in ["up", "down", "left", "right"]:
            self._move_label_3d(key)
        elif key == "rotate_cw":
            self._rotate_label_3d(angle_deg=-5)
        elif key == "rotate_ccw":
            self._rotate_label_3d(angle_deg=+5)
        

    # ---------------- Move label in 3D ----------------
    def _move_label_3d(self, arrow_key):
        label = self.label_panel.selected_label
        rgba = self.seg_rgba
        H, W, D, _ = rgba.shape
        
        # Get RGBA tuple for this label (e.g., (255,0,0,255))
        color = self.label_colors[label]
        color = color + (1.0,)  # add alpha=1.0
        color = np.array(color, dtype=np.float32)

        # Step 1 — find voxels of this label (3-D coordinates)
        mask = np.all(rgba == color, axis=-1)
        coords = np.array(np.where(mask)).T   # shape (N,3)
        if coords.size == 0:
            return

        # Step 2 — compute translation vector
        dx = dy = dz = 0
        
        if self.focused_canvas.title == "axial":
            if arrow_key == "up":    dy = +1
            if arrow_key == "down":  dy = -1
            if arrow_key == "left":  dx = +1
            if arrow_key == "right": dx = -1

        elif self.focused_canvas.title == "coronal":
            if arrow_key == "up":    dz = +1
            if arrow_key == "down":  dz = -1
            if arrow_key == "left":  dx = +1
            if arrow_key == "right": dx = -1

        elif self.focused_canvas.title == "sagittal":
            if arrow_key == "up":    dz = +1
            if arrow_key == "down":  dz = -1
            if arrow_key == "left":  dy = +1
            if arrow_key == "right": dy = -1

        # # Step 3 — erase old positions
        rgba[mask] = [0, 0, 0, 0]   # or transparent background

        # # Step 4 — translate coordinates
        new_coords = coords + np.array([dx, dy, dz])

        # # Clamp to boundaries
        new_coords[:,0] = np.clip(new_coords[:,0], 0, H-1)
        new_coords[:,1] = np.clip(new_coords[:,1], 0, W-1)
        new_coords[:,2] = np.clip(new_coords[:,2], 0, D-1)

        # # Step 5 — write label color to translated voxels
        rgba[new_coords[:,0], new_coords[:,1], new_coords[:,2]] = color

        # # Save modified volume
        self.seg_rgba = rgba

        # # Step 6 — update all canvases
        self._update_all()

    def _resize_label_3d(self, sign):
        """
        sign = '+' expand (dilate)
        sign = '-' shrink (erode)
        """

        label = self.label_panel.selected_label
        rgba = self.seg_rgba.copy()
        H, W, D, _ = rgba.shape

        # RGBA color tuple for this label
        color = np.array(self.label_colors[label] + (1.0,), dtype=np.float32)

        # Step 1 — get 3D mask
        mask = np.all(rgba == color, axis=-1)   # shape (H,W,D)
        if not np.any(mask):
            return

        # Step 2 — morphological op
        if sign == "+":
            new_mask = binary_dilation(mask, iterations=1)
        else:  # sign == "-"
            new_mask = binary_erosion(mask, iterations=1)

        # Step 3 — erase old voxels
        rgba[mask] = [0,0,0,0]

        # Step 4 — write new voxels
        rgba[new_mask] = color

        # Step 5 — update state
        self.seg_rgba = rgba
        self._update_all()

    def _rotate_label_3d(self, angle_deg=5):
        axis = {"axial":"z", "coronal":"y", "sagittal":"x"}[self.focused_canvas.title]
        if axis == "x":
            rotated = rotate(self.seg_rgba, angle_deg, axes=(1,2), reshape=False, order=0, mode='constant', cval=0,)
        elif axis == "y":
            rotated = rotate(self.seg_rgba, angle_deg, axes=(0,2), reshape=False, order=0, mode='constant', cval=0,)
        else: # z
            rotated = rotate(self.seg_rgba, angle_deg, axes=(0,1), reshape=False, order=0, mode='constant', cval=0,)

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
        self.pos = [s // 2 for s in self.shape]  # initial crosshair at center
        self.is_rgb = (self.volume.ndim == 4 and self.volume.shape[-1] == 3)

        # update slider ranges
        self.axial_slider.setRange(0, self.shape[2] - 1)
        self.coronal_slider.setRange(0, self.shape[1] - 1)
        self.sagittal_slider.setRange(0, self.shape[0] - 1)
        self.axial_slider.setValue(self.pos[2])
        self.coronal_slider.setValue(self.pos[1])
        self.sagittal_slider.setValue(self.pos[0])
        self.axial_label.setText(str(self.pos[2] + 1))
        self.coronal_label.setText(str(self.pos[1] + 1))
        self.sagittal_label.setText(str(self.pos[0] + 1))

        self._update_all()

    def _load_segmentation_nifti(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Segmentation", "", "NIfTI (*.nii *.nii.gz)")
        if not path:
            return
        
        data, aff = load_volume(path)
        assert data.shape == self.volume.shape, f"Shape mismatch: {data.shape} vs {self.volume.shape}"
        np.testing.assert_array_equal(aff, self.affine, "Affine is not equal to main nifti file's Affine !")
        self.seg_volume = data.astype(int)
        labels = np.unique(self.seg_volume)
        rng = np.random.default_rng(0)
        self.label_colors = {
            l: tuple(rng.random(3)) for l in labels if l != 0
        }
        self.label_panel.set_labels(self.label_colors)
        # PRECOMPUTE RGBA SEGMENTATION
        self._label_to_rgba(self.seg_volume)
        self._update_all()

    def _reload_segmentation_nifti(self):
        self._label_to_rgba(self.seg_volume)
        self._update_all()

    def _label_to_rgba(self, seg_volume:np.ndarray) -> None:
        h, w, d = seg_volume.shape
        self.seg_rgba = np.zeros((h, w, d, 4), dtype=np.float32)

        for l, color in self.label_colors.items():
            mask = (seg_volume == l)
            self.seg_rgba[mask, :3] = color
            self.seg_rgba[mask,  3] = 1.0   # alpha = 1 initially

    def _rgba_to_label(self, rgba_volume) -> np.ndarray:
        seg = np.zeros(rgba_volume.shape[:3], dtype=np.int32)

        for label, color in self.label_colors.items():
            rgba_color = np.array(color + (1.0,), dtype=np.float32) # add alpha=1.0
            mask = np.all(rgba_volume == rgba_color, axis=-1)
            seg[mask] = label

        return seg

    def _save_segmentation_nifti(self):
        if self.seg_rgba is None:
            return
        path, x = QtWidgets.QFileDialog.getSaveFileName(self, "Save Segmentation", "", "NIfTI (*.nii.gz *.nii)")
        print(x)
        print(path)
        if not path:
            return
        seg_img = self._rgba_to_label(self.seg_rgba)
        img = nib.Nifti1Image(seg_img, affine=self.affine)
        nib.save(img, path)

    def _toggle_seg_visibility(self, state):
        self._update_all()

    def _change_opacity(self, value):
        self._update_all()

    def _toggle_zoom_mode(self, is_enabled: bool):
        for c in (self.axial_canvas, self.coronal_canvas, self.sagittal_canvas):
            c.zoom_enabled = is_enabled

    def _toggle_zoom_mode_checkbox(self, state):
        enabled = (state == QtCore.Qt.Checked)
        self._toggle_zoom_mode(enabled)

    def _test_init(self):
        vol, aff = load_volume("./subject_001_T1_native_restored.nii.gz")
        self.volume = vol.astype(float)
        self.affine = aff
        self.shape = self.volume.shape[:3]
        self.pos = [s // 2 for s in self.shape]
        self.is_rgb = (self.volume.ndim == 4 and self.volume.shape[-1] == 3)

        # update slider ranges
        self.axial_slider.setRange(0, self.shape[2] - 1)
        self.coronal_slider.setRange(0, self.shape[1] - 1)
        self.sagittal_slider.setRange(0, self.shape[0] - 1)
        self.axial_slider.setValue(self.pos[2])
        self.coronal_slider.setValue(self.pos[1])
        self.sagittal_slider.setValue(self.pos[0])
        self.axial_label.setText(str(self.pos[2] + 1))
        self.coronal_label.setText(str(self.pos[1] + 1))
        self.sagittal_label.setText(str(self.pos[0] + 1))

        seg, aff = load_volume("./subject_001_T1_native_structures_labeled.nii.gz")
        self.seg_volume = seg.astype(int)
        labels = np.unique(self.seg_volume)
        rng = np.random.default_rng(0)
        self.label_colors = {
            l: tuple(rng.random(3)) for l in labels if l != 0
        }
        self.label_panel.set_labels(self.label_colors)
        # PRECOMPUTE RGBA SEGMENTATION
        h, w, d = self.seg_volume.shape
        self.seg_rgba = np.zeros((h, w, d, 4), dtype=np.float32)

        for l, color in self.label_colors.items():
            mask = (self.seg_volume == l)
            self.seg_rgba[mask, :3] = color
            self.seg_rgba[mask,  3] = 1.0   # alpha = 1 initially
        self._update_all()


def main():
    app = QtWidgets.QApplication(sys.argv)
    viewer = ViewerApp()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
