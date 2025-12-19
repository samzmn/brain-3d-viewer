import sys
import os
import json
from typing import Tuple

import numpy as np
import nibabel as nib
from scipy.ndimage import center_of_mass, binary_dilation, binary_erosion, rotate
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseEvent
import matplotlib.patches as patches

from utils import load_volume, label_structure, upsample_slice
import enhance


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

    def set_labels(self, seg_colors, seg_names=None):
        self.label_list.clear()
        for index, color in seg_colors.items():
            if seg_names is not None:
                label_name = seg_names[index]["name"]
                item = QtWidgets.QListWidgetItem(f"{index} . {label_name}")
            else:
                item = QtWidgets.QListWidgetItem(f"Label {index}")
            pix = QtGui.QPixmap(20,20)
            pix.fill(QtGui.QColor(*(int(c*255) for c in color)))
            item.setIcon(QtGui.QIcon(pix))
            item.setData(QtCore.Qt.UserRole, index)
            self.label_list.addItem(item)

    def on_item_clicked(self, item):
        label_idx = item.data(QtCore.Qt.UserRole)
        self.current_label_display.setText(f"Selected Label: {label_idx}")
        self.label_selected.emit(label_idx)
        
class SliceCanvas(FigureCanvas):
    """A matplotlib canvas showing a single 2D slice with a red crosshair."""
    def __init__(self, parent=None, title=""):
        self.fig = Figure(figsize=(16, 16), dpi=100)
        # dpi = 100
        # figsize = self._compute_figsize(
        #     shape=(512, 512),  # placeholder, updated later
        #     scale=1.0,
        #     dpi=dpi
        # )
        # self.fig = Figure(figsize=figsize, dpi=dpi)
        self.fig.tight_layout()
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(title)
        self.title = title
        self.shape = None
        self.aspect = 'auto'
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

        self.brush_cursor = patches.Circle(
            (0, 0),
            radius=5,
            facecolor=(1, 0, 0, 0.25),
            edgecolor=(1, 0, 0, 0.6),
            linewidth=1.5,
            visible=False,
            zorder=999
        )
        self.ax.add_patch(self.brush_cursor)
    
    def draw_focus_border(self):
        self.focus_rect.set_visible(self.is_focused)
        self.draw_idle()

    def show_empty(self):
        if self.im is not None:
            self.im.remove()
            self.im = None
        if self.seg is not None:
            self.seg.remove()
            self.seg = None
        self.draw_idle()

    def show_slice(self, slice2d, seg_slice2d=None, extend=None):
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
            self.im = self.ax.imshow(slice2d, cmap=cmap, aspect=self.aspect, origin='lower', interpolation='nearest', extent=extend)
        else:
            self.im.set_data(slice2d)
            if cmap:
                self.im.set_clim(np.nanmin(slice2d), np.nanmax(slice2d))
        
        if self.seg is None:
            if seg_slice2d is not None:
                self.seg = self.ax.imshow(seg_slice2d, aspect=self.aspect, origin='lower', interpolation='nearest', extent=extend)
        else:
            if seg_slice2d is not None:
                self.seg.set_data(seg_slice2d)
            else:
                self.seg.remove()
                self.seg = None

        self.draw_idle()

    def set_crosshair(self, x, y):
        # x0, x1 = self.ax.get_xlim()
        # y0, y1 = self.ax.get_ylim()

        # x = np.clip(x, x0, x1)
        # y = np.clip(y, y0, y1)
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

    def _on_press(self, event: MouseEvent):
        if event.inaxes != self.ax:
            return
        self.pressed = True
        if self.on_drag and not self.zoom_enabled:
            self.on_drag(event.xdata, event.ydata)
        self.parent().parent().canvas_clicked(self)  # notify main app
        self.prev_drag = (event.xdata, event.ydata)  # for panning

    def _on_move(self, event: MouseEvent):
        if event.inaxes != self.ax:
            self.brush_cursor.set_visible(False)
            self.draw_idle()
            return

        app = self.parent().parent()

        if not self.pressed:
            if not app.brush_enabled:
                self.brush_cursor.set_visible(False)
                self.draw_idle()
                return

            self.brush_cursor.center = (event.xdata, event.ydata)
            self.brush_cursor.radius = (app.brush_radius - 0.5) * app.upsample_factor
            self.brush_cursor.set_visible(True)
            self.draw_idle()

        else: # Mouse Drag
            if self.zoom_enabled:
                self._on_pan(event)
            elif app.brush_enabled:
                self.brush_cursor.center = (event.xdata, event.ydata)
                app._apply_brush(self, event)
            else:
                if self.on_drag:
                    self.on_drag(event.xdata, event.ydata)

    def _on_release(self, event: MouseEvent):
        self.pressed = False
        self.prev_drag = None
        app = self.parent().parent()
        if app.brush_enabled and not self.zoom_enabled:
            app._apply_brush(self, event)

    def _on_scroll(self, event: MouseEvent):
        step = 1 if event.button == 'up' else -1
        if self.zoom_enabled:
            self._on_zoom(step, event)
        else:
            if self.on_scroll is not None:
                self.on_scroll(step) # normal callback to viewer

    def _on_pan(self, event: MouseEvent):
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

    def _on_zoom(self, step, event: MouseEvent):
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

    def set_brush_color(self, rgba):
        if hasattr(self, "brush_cursor"):
            self.brush_cursor.set_facecolor(rgba)
            r, g, b, _ = rgba
            self.brush_cursor.set_edgecolor((r, g, b, 0.8))
            self.draw_idle()

    # def _compute_figsize(self, shape, scale, dpi, max_inches=32):
    #     H, W = shape
    #     H *= scale
    #     W *= scale

    #     fig_w = W / dpi
    #     fig_h = H / dpi

    #     # constrain to reasonable screen size
    #     scale_down = max(fig_w / max_inches, fig_h / max_inches, 1.0)

    #     return fig_w / scale_down, fig_h / scale_down

    # def update_figure_size(self, slice_shape, upsample_factor):
    #     dpi = self.fig.get_dpi()

    #     fig_w, fig_h = self._compute_figsize(
    #         slice_shape,
    #         upsample_factor,
    #         dpi
    #     )

    #     self.fig.set_size_inches(fig_w, fig_h, forward=True)
    #     self.draw_idle()

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
        self.active_label = None
        self.label_colors = {}
        self.structures = None # dict of mapping from label index to its prperty dict
        self.focused_canvas = None
        self.t1_volume = None
        self.t1_affine = None
        self.t2_volume = None
        self.t2_affine = None
        self.current_modality = "T1"   # "T1" or "T2"
        self.brainmask = None
        self.second_rio_mask = None
        
        self.brush_enabled = False
        self.brush_color = (1.0, 0.0, 0.0, 0.25)  # default RGBA
        self.brush_radius = 1
        self.brush_mode = "paint"  # or "erase"

        self.upsample_enabled = True
        self.upsample_factor = 2.

        self._init_toolbar()
        self._init_layout()
        # self._test_init() # initialize images

    def _init_toolbar(self):
        # ----------------------- TOP TOOL BAR -----------------------
        toolbar = QtWidgets.QToolBar("MainToolbar")
        self.addToolBar(QtCore.Qt.TopToolBarArea, toolbar)

        load_btn = QtWidgets.QAction("Load T1 NIfTI...", self)
        load_btn.triggered.connect(self._load_t1_volume)
        toolbar.addAction(load_btn)

        toolbar.addSeparator()

        load_t2_btn = QtWidgets.QAction("Load T2 NIfTI...", self)
        load_t2_btn.triggered.connect(self._load_t2_volume)
        toolbar.addAction(load_t2_btn)

        toolbar.addSeparator()

        self.modality_group = QtWidgets.QButtonGroup(self)
        self.t1_radio = QtWidgets.QRadioButton("T1")
        self.t2_radio = QtWidgets.QRadioButton("T2")
        self.t1_radio.setChecked(True)
        self.modality_group.addButton(self.t1_radio)
        self.modality_group.addButton(self.t2_radio)
        self.t1_radio.toggled.connect(self._change_modality)
        self.t1_radio.setDisabled(True)
        self.t2_radio.setDisabled(True)
        toolbar.addWidget(self.t1_radio)
        toolbar.addWidget(self.t2_radio)

        toolbar.addSeparator()

        load_seg_btn = QtWidgets.QAction("Load Segmentation NIfTI...", self)
        load_seg_btn.triggered.connect(self._load_segmentation_nifti)
        toolbar.addAction(load_seg_btn)

        toolbar.addSeparator()

        save_seg_btn = QtWidgets.QAction("Save Segmentation NIfTI...", self)
        save_seg_btn.triggered.connect(self._save_segmentation_nifti)
        toolbar.addAction(save_seg_btn)

        toolbar.addSeparator()

        reload_seg_btn = QtWidgets.QAction("Reload Segmentation", self)
        reload_seg_btn.triggered.connect(self._reload_segmentation_nifti)
        toolbar.addAction(reload_seg_btn)

        toolbar.addSeparator()

        self.seg_checkbox = QtWidgets.QCheckBox("Show Segmentation")
        self.seg_checkbox.setChecked(True)
        self.seg_checkbox.stateChanged.connect(self._toggle_seg_visibility)
        toolbar.addWidget(self.seg_checkbox)

        toolbar.addSeparator()

        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setSingleStep(10)
        self.opacity_slider.setPageStep(10)
        self.opacity_slider.valueChanged.connect(self._change_opacity)
        toolbar.addWidget(QtWidgets.QLabel("Opacity of Segmentation:"))
        toolbar.addWidget(self.opacity_slider)

        toolbar.addSeparator()

        self.zoom_checkbox = QtWidgets.QCheckBox("Zoom Mode")
        self.zoom_checkbox.setChecked(False)
        self.zoom_checkbox.stateChanged.connect(self._toggle_zoom_mode_checkbox)
        toolbar.addWidget(self.zoom_checkbox)

        toolbar.addSeparator()

        reset_btn = QtWidgets.QAction("Close all files", self)
        reset_btn.triggered.connect(self._reset_volumes)
        toolbar.addAction(reset_btn)

        # Second toolbar ----------------------------
        self.addToolBarBreak()            # force new row
        toolbar2 = QtWidgets.QToolBar("SecondaryToolbar")
        self.addToolBar(QtCore.Qt.TopToolBarArea, toolbar2)

        self.filter_checkbox = QtWidgets.QCheckBox("Apply Filter ")
        self.filter_checkbox.setChecked(False)
        self.filter_checkbox.stateChanged.connect(self._toggle_filter_checkbox)
        toolbar2.addWidget(self.filter_checkbox)

        toolbar2.addSeparator()

        self.contrast_checkbox = QtWidgets.QCheckBox("Contrast: ")
        self.contrast_checkbox.setChecked(False)
        self.contrast_checkbox.stateChanged.connect(self._update_all)
        self.contrast_checkbox.stateChanged.connect(self._reset_default_filter_values)
        toolbar2.addWidget(self.contrast_checkbox)

        self.contrast_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.contrast_slider.setMinimum(0)
        self.contrast_slider.setMaximum(50)
        self.contrast_slider.setValue(12)
        self.contrast_slider.setSingleStep(1)
        self.contrast_slider.valueChanged.connect(self._on_clahe_slider_change)
        toolbar2.addWidget(QtWidgets.QLabel(""))
        toolbar2.addWidget(self.contrast_slider)

        toolbar2.addSeparator()

        self.denoise_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.denoise_slider.setMinimum(0)
        self.denoise_slider.setMaximum(100)
        self.denoise_slider.setValue(60)
        self.denoise_slider.setSingleStep(5)
        self.denoise_slide_label = QtWidgets.QLabel("Denoise: 0.6  ")
        self.denoise_slider.valueChanged.connect(self._update_all)
        self.denoise_slider.valueChanged.connect(lambda value: self.denoise_slide_label.setText(f"Denoise: {value/100:.1f}  "))
        toolbar2.addWidget(self.denoise_slide_label)
        toolbar2.addWidget(self.denoise_slider)

        toolbar2.addSeparator()

        self.upsample_checkbox = QtWidgets.QCheckBox("High Resolution: ")
        self.upsample_checkbox.setChecked(True)
        self.upsample_checkbox.stateChanged.connect(self._toggle_upsample_checkbox)
        toolbar2.addWidget(self.upsample_checkbox)

        self.upsample_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.upsample_slider.setMinimum(2)
        self.upsample_slider.setMaximum(4)
        self.upsample_slider.setValue(2)
        self.upsample_slider.setSingleStep(1)
        self.upsample_slider.setMaximumWidth(200)
        self.upsample_slider.valueChanged.connect(self._upsample_slider_change)
        self.upsample_slider_label = QtWidgets.QLabel(f" X {self.upsample_factor}  ")
        self.upsample_slider.valueChanged.connect(lambda value: self.upsample_slider_label.setText(f" X {self.upsample_factor}  "))
        toolbar2.addWidget(self.upsample_slider_label)
        toolbar2.addWidget(self.upsample_slider)        

        # Third toolbar ----------------------------
        toolbar3 = QtWidgets.QToolBar("Brush Tools", self)
        self.addToolBarBreak()
        self.addToolBar(QtCore.Qt.TopToolBarArea, toolbar3)

        self.brush_checkbox = QtWidgets.QCheckBox("Brush")
        self.brush_checkbox.toggled.connect(self._toggle_brush)
        toolbar3.addWidget(self.brush_checkbox)

        toolbar3.addSeparator()

        self.brush_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brush_slider.setMaximumWidth(200)
        self.brush_slider.setRange(1, 10)
        self.brush_slider.setValue(self.brush_radius)
        self.brush_slider.valueChanged.connect(self._set_brush_radius)
        self.brush_slider_label = QtWidgets.QLabel(f"Size: {self.brush_radius}  ")
        self.brush_slider.valueChanged.connect(lambda value: self.brush_slider_label.setText(f"Size: {value}  "))
        toolbar3.addWidget(self.brush_slider_label)
        toolbar3.addWidget(self.brush_slider)

        toolbar3.addSeparator()

        self.brush_radio = QtWidgets.QRadioButton("Paint")
        self.erase_radio = QtWidgets.QRadioButton("Erase")
        self.brush_radio.setChecked(True)

        self.brush_radio.toggled.connect(lambda: self._set_brush_mode("paint"))
        self.erase_radio.toggled.connect(lambda: self._set_brush_mode("erase"))

        toolbar3.addWidget(self.brush_radio)
        toolbar3.addWidget(self.erase_radio)

    def _init_layout(self):
        # ----------------------- MAIN WIDGET and layout -----------------------
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # ---------------- Left panel (labels) ----------------------
        self.label_panel = LabelPanel()
        self.label_panel.label_selected.connect(self._on_label_selected)
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
        self.axial_canvas_2 = SliceCanvas(self, title="")

        for c in (self.axial_canvas, self.coronal_canvas, self.sagittal_canvas):
            c.enable_interaction()  

        self.axial_canvas.on_drag = self._on_axial_drag
        self.coronal_canvas.on_drag = self._on_coronal_drag
        self.sagittal_canvas.on_drag = self._on_sagittal_drag

        self.axial_canvas.on_scroll = self._on_axial_scroll
        self.coronal_canvas.on_scroll = self._on_coronal_scroll
        self.sagittal_canvas.on_scroll = self._on_sagittal_scroll

        QtWidgets.QApplication.instance().installEventFilter(self)

        # ----------- layout with label + slider + maximize button -----------
        self._rebuild_grid_layout()

        self.move(100, 100)
        self.showMaximized()

        # status bar text
        self.status = self.statusBar()
        self._update_status()

    # ---------------- Status bar update ----------------
    def _update_status(self):
        if self.pos is not None:
            self.status.showMessage(f'pos (x,y,z): {self.pos[0]}, {self.pos[1]}, {self.pos[2]}')

    # ---------------- Visual Filters Toolbar callbaccks -----------
    def _reset_default_filter_values(self):
        if self.current_modality == "T1":
            self.contrast_slider.setValue(20)
            self.denoise_slider.setValue(40)
        else:
            self.contrast_slider.setValue(15)
            self.denoise_slider.setValue(60)

    def _on_clahe_slider_change(self, value):
        self._update_all()

    # ---------------- 2D slices ----------------
    def _make_seg_overlay(self, seg2d):
        """Given a seg2d RGBA slice, apply opacity and return overlay."""
        overlay = seg2d.copy()
        overlay[..., 3] *= (self.opacity_slider.value() / 100.0) # Alpha
        overlay = np.transpose(overlay, (1, 0, 2))
        overlay = np.fliplr(overlay)
        return overlay

    def _get_normal_axial(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.axial_canvas_2.ax is not None:
            self.axial_canvas_2.ax.set_xlim((1 / self.upsample_factor) * np.array(self.axial_canvas.ax.get_xlim(), dtype=np.float64))
            self.axial_canvas_2.ax.set_ylim((1 / self.upsample_factor) * np.array(self.axial_canvas.ax.get_ylim(), dtype=np.float64))
        slice2d = self.volume[:, :, self.pos[2]]
        slice2d = np.fliplr(slice2d.T)
        return slice2d, None
    
    def _get_axial(self) -> Tuple[np.ndarray, np.ndarray]:
        slice2d = self.volume[:, :, self.pos[2], :] if self.is_rgb else self.volume[:, :, self.pos[2]]
        # slice2d = axial_slab_average(self.volume.copy(), self.pos[2])
        slice2d = np.fliplr(slice2d.T)  # transpose for correct orientation
        if self.upsample_enabled:
            slice2d = upsample_slice(slice2d, self.upsample_factor, order=1)

        if self.contrast_checkbox.isChecked() and self.filter_checkbox.isChecked():
            roi_mask = self.second_roi_mask[:, :, self.pos[2]]
            roi_mask = np.fliplr(roi_mask.T)
            if self.upsample_enabled:
                roi_mask = upsample_slice(roi_mask, self.upsample_factor, order=0)
            denoise_value = self.denoise_slider.value() / 100
            slice2d = enhance.roi_denoise(slice2d, roi_mask, sigma=denoise_value)
            if self.current_modality == "T1":
                contrast_value = self.contrast_slider.value()
                slice2d = enhance.roi_window_tighten(slice2d, roi_mask, low_pct=contrast_value, high_pct=100 - contrast_value)
                # slice2d = enhance.roi_clahe(slice2d, roi_mask, kernel_size=16, clip_limit=0.01, blend=0.3)
                # slice2d = enhance.roi_dog_enhance(slice2d, roi_mask, sigma_low=0.6, sigma_high=1.2, amount=0.25)
            else:
                contrast_value = self.contrast_slider.value() / 1000
                slice2d = enhance.roi_clahe(slice2d, roi_mask, kernel_size=24, clip_limit=contrast_value, blend=0.5)
                slice2d = enhance.roi_window_tighten(slice2d, roi_mask, low_pct=1, high_pct=99)

        if self.seg_rgba is not None and self.seg_checkbox.isChecked():
            seg_slice2d = self._make_seg_overlay(self.seg_rgba[:, :, self.pos[2]])
            if self.upsample_enabled:
                seg_slice2d = upsample_slice(seg_slice2d, (self.upsample_factor,
                                            self.upsample_factor,
                                            1),
                                order=0)
        else:
            seg_slice2d = None
        return slice2d, seg_slice2d

    def _get_coronal(self) -> Tuple[np.ndarray, np.ndarray]:
        slice2d = self.volume[:, self.pos[1], :, :] if self.is_rgb else self.volume[:, self.pos[1], :]
        slice2d = np.fliplr(slice2d.T)  # transpose for correct orientation
        if self.upsample_enabled:
            slice2d = upsample_slice(slice2d, self.upsample_factor, order=1)

        if self.contrast_checkbox.isChecked() and self.filter_checkbox.isChecked():
            roi_mask = self.second_roi_mask[:, self.pos[1], :]
            roi_mask = np.fliplr(roi_mask.T)
            if self.upsample_enabled:
                roi_mask = upsample_slice(roi_mask, self.upsample_factor, order=0)
            denoise_value = self.denoise_slider.value() / 100
            slice2d = enhance.roi_denoise(slice2d, roi_mask, sigma=denoise_value)
            if self.current_modality == "T1":
                contrast_value = self.contrast_slider.value()
                slice2d = enhance.roi_window_tighten(slice2d, roi_mask, low_pct=contrast_value, high_pct=100 - contrast_value)
                # slice2d = enhance.roi_clahe(slice2d, roi_mask, kernel_size=16, clip_limit=0.01, blend=0.3)
                # slice2d = enhance.roi_dog_enhance(slice2d, roi_mask, sigma_low=0.6, sigma_high=1.2, amount=0.25)
            else:
                contrast_value = self.contrast_slider.value() / 1000
                slice2d = enhance.roi_clahe(slice2d, roi_mask, kernel_size=24, clip_limit=contrast_value, blend=0.5)
                slice2d = enhance.roi_window_tighten(slice2d, roi_mask, low_pct=1, high_pct=99)

        if self.seg_rgba is not None and self.seg_checkbox.isChecked():
            seg_slice2d = self._make_seg_overlay(self.seg_rgba[:, self.pos[1], :])
            if self.upsample_enabled:
                seg_slice2d = upsample_slice(seg_slice2d, (self.upsample_factor,
                                            self.upsample_factor,
                                            1),
                                order=0)
        else:
            seg_slice2d = None
        return slice2d, seg_slice2d

    def _get_sagittal(self) -> Tuple[np.ndarray, np.ndarray]:
        slice2d = self.volume[self.pos[0], :, :, :] if self.is_rgb else self.volume[self.pos[0], :, :]
        slice2d = np.fliplr(slice2d.T)  # transpose for correct orientation
        if self.upsample_enabled:
            slice2d = upsample_slice(slice2d, self.upsample_factor, order=1)

        if self.contrast_checkbox.isChecked() and self.filter_checkbox.isChecked():
            roi_mask = self.second_roi_mask[self.pos[0], :, :]
            roi_mask = np.fliplr(roi_mask.T)
            if self.upsample_enabled:
                roi_mask = upsample_slice(roi_mask, self.upsample_factor, order=0)
            denoise_value = self.denoise_slider.value() / 100
            slice2d = enhance.roi_denoise(slice2d, roi_mask, sigma=denoise_value)
            if self.current_modality == "T1":
                contrast_value = self.contrast_slider.value()
                slice2d = enhance.roi_window_tighten(slice2d, roi_mask, low_pct=contrast_value, high_pct=100 - contrast_value)
                # slice2d = enhance.roi_clahe(slice2d, roi_mask, kernel_size=16, clip_limit=0.01, blend=0.3)
                # slice2d = enhance.roi_dog_enhance(slice2d, roi_mask, sigma_low=0.6, sigma_high=1.2, amount=0.25)
            else:
                contrast_value = self.contrast_slider.value() / 1000
                slice2d = enhance.roi_clahe(slice2d, roi_mask, kernel_size=24, clip_limit=contrast_value, blend=0.5)
                slice2d = enhance.roi_window_tighten(slice2d, roi_mask, low_pct=1, high_pct=99)

        if self.seg_rgba is not None and self.seg_checkbox.isChecked():
            seg_slice2d = self._make_seg_overlay(self.seg_rgba[self.pos[0], :, :])
            if self.upsample_enabled:
                seg_slice2d = upsample_slice(seg_slice2d, (self.upsample_factor,
                                            self.upsample_factor,
                                            1),
                                order=0)
        else:
            seg_slice2d = None
        return slice2d, seg_slice2d

    def _voxel_to_display(self, v):
        return (v + 0.5) * self.upsample_factor

    def _update_all(self):
        # update 2D images
        self.axial_canvas.show_slice(*self._get_axial())
        self.coronal_canvas.show_slice(*self._get_coronal())
        self.sagittal_canvas.show_slice(*self._get_sagittal())
        # self.axial_canvas_2.show_slice(*self._get_normal_axial())

        # update crosshairs: compute pixel coords for each canvas
        self.axial_canvas.set_crosshair(
            self._voxel_to_display(self.shape[0] - 1 - self.pos[0]),
            self._voxel_to_display(self.pos[1])
        )

        self.coronal_canvas.set_crosshair(
            self._voxel_to_display(self.shape[0] - 1 - self.pos[0]),
            self._voxel_to_display(self.pos[2])
        )

        self.sagittal_canvas.set_crosshair(
            self._voxel_to_display(self.shape[1] - 1 - self.pos[1]),
            self._voxel_to_display(self.pos[2])
        )

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

        self.axial_label.setText(str(self.pos[2] + 1))
        self.coronal_label.setText(str(self.pos[1] + 1))
        self.sagittal_label.setText(str(self.pos[0] + 1))

    # ---------------- Canvas click handling ----------------
    def canvas_clicked(self, canvas):
        for c in [self.axial_canvas, self.sagittal_canvas, self.coronal_canvas]:
            c.is_focused = (c is canvas)
            c.draw_focus_border()
        self.focused_canvas = canvas

    # ---------------- Drag callbacks ----------------
    def _on_axial_drag(self, xpix: float, ypix: float):
        if xpix is None or ypix is None:
            return
        x = int(xpix / self.upsample_factor)
        y = int(ypix / self.upsample_factor)
        self.pos[0] = np.clip(self.shape[0] - 1 - x, 0, self.shape[0] - 1)
        self.pos[1] = np.clip(y, 0, self.shape[1] - 1)
        self._update_all()

    def _on_coronal_drag(self, xpix: float, ypix: float):
        if xpix is None or ypix is None:
            return
        x = int(xpix / self.upsample_factor)
        y = int(ypix / self.upsample_factor)
        self.pos[0] = np.clip(self.shape[0] - 1 - x, 0, self.shape[0] - 1)
        self.pos[2] = np.clip(y, 0, self.shape[2] - 1)
        self._update_all()

    def _on_sagittal_drag(self, xpix: float, ypix: float):
        if xpix is None or ypix is None:
            return
        x = int(xpix / self.upsample_factor)
        y = int(ypix / self.upsample_factor)
        self.pos[1] = np.clip(self.shape[1] - 1 - x, 0, self.shape[1] - 1)
        self.pos[2] = np.clip(y, 0, self.shape[2] - 1)
        self._update_all()

    # ---------------- Scroll callbacks ----------------
    def _on_axial_scroll(self, step):
        # axial = Z axis slice
        self.pos[2] = np.clip(self.pos[2] + step, 0, self.shape[2]-1)
        self._update_all()

    def _on_coronal_scroll(self, step):
        # coronal = Y axis slice
        self.pos[1] = np.clip(self.pos[1] + step, 0, self.shape[1]-1)
        self._update_all()

    def _on_sagittal_scroll(self, step):
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

        axial_row_2 = QtWidgets.QHBoxLayout()
        grid.addLayout(axial_row_2, 2, 1)

        # 3) Re-add canvases
        grid.addWidget(self.axial_canvas,    1, 0)
        grid.addWidget(self.coronal_canvas,  1, 1)
        grid.addWidget(self.sagittal_canvas, 3, 0)
        grid.addWidget(self.axial_canvas_2,  3, 1)

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

    # --------------- Left Label Panel callbacks -------------------
    def _on_label_selected(self, label_idx):
        self.active_label = label_idx

        if label_idx in self.label_colors:
            r, g, b = self.label_colors[self.active_label]
            self.active_label_color = (r, g, b, 0.25)
        else:
            self.active_label_color = (1.0, 0.0, 0.0, 0.25)

        for canvas in (self.axial_canvas, self.coronal_canvas, self.sagittal_canvas):
            canvas.set_brush_color(self.active_label_color)

    # ---------------- Key press handling ----------------
    def eventFilter(self, obj, event):
        # ---------------- Key press ----------------
        if event.type() == QtCore.QEvent.KeyPress:
            modifiers = event.modifiers()
            # CTRL → zoom mode
            if modifiers & QtCore.Qt.ControlModifier:
                self._toggle_zoom_mode(True)
            else:
                self._toggle_zoom_mode(False)

            # SHIFT → brush mode (momentary)
            if modifiers & QtCore.Qt.ShiftModifier:
                self._toggle_brush_mode(True)
            else:
                self._toggle_brush_mode(False)

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

        # ---------------- Key release ----------------
        elif event.type() == QtCore.QEvent.KeyRelease:

            modifiers = event.modifiers()

            # CTRL released → disable zoom
            if not (modifiers & QtCore.Qt.ControlModifier):
                self._toggle_zoom_mode(False)

            # SHIFT released → disable brush
            if not (modifiers & QtCore.Qt.ShiftModifier):
                self._toggle_brush_mode(False)

        return super().eventFilter(obj, event)
    
    def _on_key_press(self, key):
        if self.active_label is None:
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
        
    # ---------------- Label manipulation in 3D ----------------
    def _move_label_3d(self, arrow_key):
        label = self.active_label
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
        sign = '+' expand based on probability threshold (preferred)
        sign = '-' shrink based on probability threshold (preferred)
        OR fallback morphological dilation/erosion.
        """
        label_idx = self.active_label
        rgba = self.seg_rgba.copy()
        H, W, D, _ = rgba.shape

        # RGBA color tuple for this label
        color = np.array(self.label_colors[label_idx] + (1.0,), dtype=np.float32)

        # Extract old mask (3D boolean mask)
        old_mask  = np.all(rgba == color, axis=-1)   # shape (H,W,D)
        if not np.any(old_mask ):
            return
        
        def place_new_structure(prob_map, threshold):
            # Build new labeled mask from probability map
            new_label_map = label_structure(prob_map, label_idx, threshold)   # uint8
            new_mask_global = new_label_map == label_idx

            if not np.any(new_mask_global):
                # nothing to place -> erase old and exit
                rgba[old_mask] = [0,0,0,0]
                self.seg_rgba = rgba
                self._update_all()
                return

            # Compute CoM BEFORE and AFTER
            old_com = center_of_mass(old_mask.astype(float))
            new_com = center_of_mass(new_mask_global.astype(float))

            # For example: old_com = (y, x, z)
            # Always convert to np.array for vector arithmetic
            old_com = np.array(old_com)
            new_com = np.array(new_com)

            # Compute shift vector to align CoMs
            shift = old_com - new_com   # vector of floats such as (dy, dx, dz)
            shift = np.round(shift).astype(int)  # voxel shift

            # Shift new mask safely into volume
            new_mask_shifted = np.zeros_like(new_mask_global, dtype=bool)

            ys, xs, zs = np.where(new_mask_global)
            ys2 = ys + shift[0]
            xs2 = xs + shift[1]
            zs2 = zs + shift[2]

            # Keep only points inside valid volume space
            valid = (
                (ys2 >= 0) & (ys2 < H) &
                (xs2 >= 0) & (xs2 < W) &
                (zs2 >= 0) & (zs2 < D)
            )
            ys2 = ys2[valid]
            xs2 = xs2[valid]
            zs2 = zs2[valid]
            new_mask_shifted[ys2, xs2, zs2] = True

            # Replace old voxels with background, write new voxels
            rgba[old_mask] = [0,0,0,0]
            rgba[new_mask_shifted] = color

            self.seg_rgba = rgba
            self._update_all()
        
        # If structure probability map is available -> use threshold method
        if self.structures is not None:
            thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
            threshold_idx = self.structures[label_idx]["threshold"]
            if threshold_idx >= 0 and threshold_idx <= 10:
                prob_map = self.structures[label_idx]["nifti"]
                if sign == "+":
                    threshold_idx -= 1
                    self.structures[label_idx]["threshold"] = threshold_idx
                    if threshold_idx >= 0:
                        place_new_structure(prob_map, threshold = thresholds[threshold_idx])
                        return
                if sign == "-":
                    threshold_idx += 1
                    self.structures[label_idx]["threshold"] = threshold_idx
                    if threshold_idx <= 10:
                        place_new_structure(prob_map, threshold = thresholds[threshold_idx])
                        return
            else:
                if sign == "+":
                    threshold_idx -= 1
                elif sign == "-":
                    threshold_idx += 1
                self.structures[label_idx]["threshold"] = threshold_idx

        # morphological op
        if sign == "+":
            new_mask = binary_dilation(old_mask, iterations=1)
        else:  # sign == "-"
            new_mask = binary_erosion(old_mask, iterations=1)

        # erase old voxels
        rgba[old_mask] = [0,0,0,0]

        # write new voxels
        rgba[new_mask] = color

        # update state
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

    # ---------------- Brush Toolbar callbacks ---------------
    def _toggle_brush_mode(self, enabled):
        self.brush_enabled = enabled
        self.brush_checkbox.setChecked(enabled)

    def _toggle_brush(self, checked):
        self.brush_enabled = checked

    def _set_brush_radius(self, value):
        self.brush_radius = value

    def _set_brush_mode(self, mode):
        self.brush_mode = mode

    def _apply_brush(self, canvas: SliceCanvas, event: MouseEvent):
        if event.xdata is None or event.ydata is None:
            return
        if self.active_label is None:
            return
        
        cx = int(round(event.xdata / self.upsample_factor))
        cy = int(round(event.ydata / self.upsample_factor))
        cx += 1 # because of a half-pixel coordinate mismatch between matplotlib and the data

        self._paint_circle(canvas, cx, cy)
        self._update_all()

    def _paint_circle(self, canvas, cx, cy):
        rr = self.brush_radius - 1
        yy, xx = np.ogrid[-rr:rr+1, -rr:rr+1]
        mask = xx**2 + yy**2 <= rr**2

        for dy in range(-rr, rr+1):
            for dx in range(-rr, rr+1):
                if not mask[dy+rr, dx+rr]:
                    continue

                x2 = cx + dx
                y2 = cy + dy

                if canvas is self.axial_canvas:
                    i = self.shape[0] - x2
                    j = y2
                    k = self.pos[2]

                elif canvas is self.coronal_canvas:
                    i = self.shape[0] - x2
                    j = self.pos[1]
                    k = y2

                elif canvas is self.sagittal_canvas:
                    i = self.pos[0]
                    j = self.shape[1] - x2
                    k = y2

                else:
                    continue

                if not self._in_bounds(i, j, k):
                    continue

                self._write_seg_voxel(i, j, k)

    def _in_bounds(self, i, j, k):
        return (
            0 <= i < self.shape[0] and
            0 <= j < self.shape[1] and
            0 <= k < self.shape[2]
        )

    def _write_seg_voxel(self, i, j, k):
        color = np.array(self.label_colors[self.active_label] + (1.0,), dtype=np.float32)
        if self.brush_mode == "paint":
            self.seg_rgba[i, j, k] = color
        else:
            self.seg_rgba[i, j, k] = (0., 0., 0., 0.)

    # ---------- Selecting Main Volume showed (T1 or T2) -----------------
    def _change_modality(self):
        if self.t1_radio.isChecked():
            if self.t1_volume is not None:
                self.current_modality = "T1"
                if self.filter_checkbox.isChecked():
                    self.volume = self._apply_volume_normalization(self.t1_volume, self.brainmask, pmin=5, pmax=95)
                else:
                    self.volume = self.t1_volume
                self.affine = self.t1_affine
        else:
            if self.t2_volume is not None:
                self.current_modality = "T2"
                if self.filter_checkbox.isChecked():
                    self.volume = self._apply_volume_normalization(self.t2_volume, self.brainmask, pmin=3, pmax=97)
                else:
                    self.volume = self.t2_volume
                self.affine = self.t2_affine

        self._update_all()

    def _toggle_filter_checkbox(self, state):
        enabled = (state == QtCore.Qt.Checked)
        if enabled:
            if self.current_modality == "T1":
                self.volume = self._apply_volume_normalization(self.t1_volume, self.brainmask, pmin=5, pmax=95)
            else:
                self.volume = self._apply_volume_normalization(self.t2_volume, self.brainmask, pmin=3, pmax=97)
        else:
            if self.current_modality == "T1":
                self.volume = self.t1_volume
            else:
                self.volume = self.t2_volume
        self._update_all()

    def _apply_volume_normalization(self, volume: np.ndarray, brainmask: np.ndarray = None, pmin=1, pmax=99) -> np.ndarray:
        if brainmask is None:
            brainmask = np.ones_like(volume, dtype=bool)
        lo, hi = enhance.compute_volume_normalization(volume, brainmask, pmin, pmax)
        return enhance.apply_volume_normalization(volume, lo, hi)

    def _toggle_upsample_checkbox(self, state):
        enabled = (state == QtCore.Qt.Checked)
        self.upsample_checkbox = enabled
        if enabled:
            self.upsample_factor = 2.
            self.upsample_slider.setValue(int(self.upsample_factor))
            self.upsample_slider.setEnabled(True)
            self.upsample_slider_label.setText(f" X {self.upsample_factor}  ")
        else:
            self.upsample_factor = 1.
            self.upsample_slider.setValue(int(self.upsample_factor))
            self.upsample_slider.setEnabled(False)
            self.upsample_slider_label.setText(f" X {self.upsample_factor}  ")

        # self._update_canvases_figures()
        self._update_all()

    def _upsample_slider_change(self, value):
        self.upsample_factor = float(value)
        # self._update_canvases_figures()
        self._update_all()

    # def _update_canvases_figures(self):
    #     if self.volume is None:
    #         return
    #     axial_slice_shape = (self.shape[1], self.shape[0])
    #     coronal_slice_shape = (self.shape[2], self.shape[0])
    #     sagittal_slice_shape = (self.shape[2], self.shape[1])
    #     self.axial_canvas.update_figure_size(axial_slice_shape, self.upsample_factor)
    #     self.coronal_canvas.update_figure_size(coronal_slice_shape, self.upsample_factor)
    #     self.sagittal_canvas.update_figure_size(sagittal_slice_shape, self.upsample_factor)

    # ---------------- Load volume ----------------
    def _load_t1_volume(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load NIfTI File", "", "NIfTI (*.nii *.nii.gz)"
        )
        if not path:
            return
        self._load_new_volume(path, modality="T1")
        self.t1_radio.setChecked(True)
        self.t1_radio.setEnabled(True)
        self.current_modality = "T1"

        patient_id = os.path.basename(path).split("_")[1]
        masks_path = os.path.join(os.path.dirname(path), "masks")
        self.brainmask = load_volume(os.path.join(masks_path, f"brain_mask_of_subject_{patient_id}.nii.gz"))[0].astype(int)
        self.second_roi_mask = load_volume(os.path.join(masks_path, f"second_subcortical_mask_of_subject_{patient_id}.nii.gz"))[0].astype(int)
        # self.third_roi_mask = load_volume(os.path.join(masks_path, f"third_subcortical_mask_of_subject_{patient_id}.nii.gz"))[0].astype(int)
        self.filter_checkbox.setChecked(True)
        self.volume = self._apply_volume_normalization(self.t1_volume, self.brainmask)
        # self.axial_volume = precompute_slab_volume(self.volume, axis=2, radius=1, sigma=0.8)
        self._update_all()

    def _load_t2_volume(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load T2 NIfTI", "", "NIfTI Files (*.nii *.nii.gz)"
        )
        if not path:
            return
        self._load_new_volume(path, modality="T2")
        self.t2_radio.setChecked(True)
        self.t2_radio.setEnabled(True)
        self.current_modality = "T2"
        self._update_all()

    def _load_new_volume(self, path: str, modality: str = "T1"):
        data, aff = load_volume(path)
        # If shape changes, update sliders etc
        if modality == "T1":
            self.t1_volume = data.astype(float)
            self.t1_affine = aff
            self.volume = self.t1_volume
            self.affine = self.t1_affine
        else:  # T2
            self.t2_volume = data.astype(float)
            self.t2_affine = aff
            self.volume = self.t2_volume
            self.affine = self.t2_affine

        if self.t1_affine is not None and self.t2_affine is not None:
            if not np.allclose(self.t1_affine, self.t2_affine):
                print("Warning: images are not aligned in world space")

        self.shape = self.volume.shape[:3]
        self.pos = [s // 2 for s in self.shape]  # initial crosshair at center
        self.is_rgb = (self.volume.ndim == 4 and self.volume.shape[-1] == 3)
        
        sx = np.linalg.norm(self.affine[:3,0])   # spacing along i axis (rows of data)
        sy = np.linalg.norm(self.affine[:3,1])   # spacing along j axis (cols of data)
        sz = np.linalg.norm(self.affine[:3,2])   # spacing along k axis (slice thickness)
        self.axial_canvas.aspect = sy / sx
        self.coronal_canvas.aspect = sz / sx
        self.sagittal_canvas.aspect = sz / sy
        self.axial_canvas_2.aspect = sy / sx

        # self._update_canvases_figures()

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
        # PRECOMPUTE RGBA SEGMENTATION
        self._label_to_rgba(self.seg_volume)

        # Read dictionary back from JSON file
        try:
            json_path = os.path.join(os.path.dirname(path), os.path.basename(path).replace("structures_labeled.nii.gz", "labels.json"))
            # print(json_path)
            with open(json_path, "r") as f:
                structures = json.load(f)
                # print(structures)
                self.structures = dict()
                for label_name, index in structures.items():
                    # print(label_name, index)
                    structure = dict()
                    structure["name"] = label_name
                    structure["threshold"] = 6
                    structure_path = os.path.join(os.path.dirname(path), "labels", label_name + "_prob_in_" + os.path.basename(path).replace("_structures_labeled", ""))
                    structure["nifti"] = self._load_segmentation_structure(structure_path)
                    self.structures[index] = structure
        except:
            print("No data.json file found for loading label names.")

        self.label_panel.set_labels(self.label_colors, self.structures)
        self._update_all()

    def _load_segmentation_structure(self, path: str):
        data, aff = load_volume(path)
        return data.astype(float)

    def _reload_segmentation_nifti(self):
        self._label_to_rgba(self.seg_volume)
        if self.structures is not None:
            for label_idx in self.structures.keys():
                self.structures[label_idx]["threshold"] = 6
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
        if is_enabled:
            self.zoom_checkbox.setChecked(True)
        else:
            self.zoom_checkbox.setChecked(False)
        for c in (self.axial_canvas, self.coronal_canvas, self.sagittal_canvas):
            c.zoom_enabled = is_enabled

    def _toggle_zoom_mode_checkbox(self, state):
        enabled = (state == QtCore.Qt.Checked)
        self._toggle_zoom_mode(enabled)

    def _reset_volumes(self):
        self.volume = None
        self.affine = None
        self.is_rgb = None
        self.shape = None
        self.pos = None
        self.seg_volume = None
        self.seg_rgba = None
        self.label_colors = {}
        self.structures = None
        self.focused_canvas = None
        self.t1_volume = None
        self.t1_affine = None
        self.t2_volume = None
        self.t2_affine = None
        self.current_modality = "T1"
        self.axial_canvas.show_empty()
        self.coronal_canvas.show_empty()
        self.sagittal_canvas.show_empty()

    def _test_init(self):
        self._load_new_volume("./test_nifti_files/001/subject_001_T1_pre.nii.gz", modality="T1")
        self._load_new_volume("./test_nifti_files/001/subject_001_T2_merged.nii.gz", modality="T2")
        self.t1_radio.setEnabled(True)
        self.t2_radio.setEnabled(True)
        self.brainmask = load_volume("./test_nifti_files/001/masks/brain_mask_of_subject_001.nii.gz")[0].astype(int)
        self.second_roi_mask = load_volume("./test_nifti_files/001/masks/second_subcortical_mask_of_subject_001.nii.gz")[0].astype(int)
        # self.third_roi_mask = load_volume("./test_nifti_files/001/masks/third_subcortical_mask_of_subject_001.nii.gz")[0].astype(int)
        self.filter_checkbox.setChecked(True)
        self.volume = self._apply_volume_normalization(self.t1_volume, self.brainmask)
        # self.axial_volume = precompute_slab_volume(self.volume, axis=2, radius=1, sigma=0.8)
        # print(50 * "-")
        # print(self.axial_volume.shape)
        seg, aff = load_volume("./test_nifti_files/001/subject_001_structures_labeled.nii.gz")
        self.seg_volume = seg.astype(int)
        labels = np.unique(self.seg_volume)
        rng = np.random.default_rng(0)
        self.label_colors = {
            l: tuple(rng.random(3)) for l in labels if l != 0
        }
        # PRECOMPUTE RGBA SEGMENTATION
        self._label_to_rgba(self.seg_volume)
        try:
            json_path = "./test_nifti_files/001/subject_001_labels.json"
            with open(json_path, "r") as f:
                structures = json.load(f)
                self.structures = dict()
                for label_name, index in structures.items():
                    structure = dict()
                    structure["name"] = label_name
                    structure["threshold"] = 6
                    structure_path = f"./test_nifti_files/001/labels/{label_name}_prob_in_subject_001.nii.gz"
                    structure["nifti"] = self._load_segmentation_structure(structure_path)
                    self.structures[index] = structure
        except:
            print("No data.json file found for loading label names.")

        self.label_panel.set_labels(self.label_colors, self.structures)
        self._update_all()


def main():
    app = QtWidgets.QApplication(sys.argv)
    viewer = ViewerApp()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
