import sys
from typing import Tuple

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseEvent
import matplotlib.patches as patches

from utils import load_volume, resource_path, fast_mri_slice_upsample
import enhance


class SliceCanvas(FigureCanvas):
    """A matplotlib canvas showing a single 2D slice with a red crosshair."""
    def __init__(self, parent=None, title=""):
        self.fig = Figure(figsize=(16, 16), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_axes([0,0,1,1])  # instead of add_subplot

        self.fig.subplots_adjust(
            left=0,
            right=1,
            bottom=0,
            top=1
        )

        self.fig.patch.set_facecolor("#000000FC")
        
        self.ax.set_title(title)
        self.title = title
        self.shape = None
        self.aspect = 'auto'
        self.im = None
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
    
    def draw_focus_border(self):
        self.focus_rect.set_visible(self.is_focused)
        self.draw_idle()

    def show_empty(self):
        if self.im is not None:
            self.im.remove()
            self.im = None
        self.draw_idle()

    def show_slice(self, slice2d, extend=None):
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

        self.draw_idle()

    def set_crosshair(self, x, y):
        if self.vline is None:
            self.vline = self.ax.axvline(
                x,
                color=(1.0, 0.4, 0.4),
                alpha=0.4,
                linestyle='--',
                linewidth=0.6
            )

            self.hline = self.ax.axhline(
                y,
                color=(1.0, 0.4, 0.4),
                alpha=0.4,
                linestyle='--',
                linewidth=0.6
            )
        else:
            self.vline.set_xdata([x,x])
            self.hline.set_ydata([y,y])
        self.draw_idle()

    def enable_interaction(self):
        self.cid_press = self.mpl_connect('button_press_event', self._on_press)
        self.cid_move = self.mpl_connect('motion_notify_event', self._on_move)
        self.cid_release = self.mpl_connect('button_release_event', self._on_release)
        self.cid_scroll = self.mpl_connect('scroll_event', self._on_scroll)

    def disable_interaction(self):
        if self.cid_press: self.mpl_disconnect(self.cid_press)
        if self.cid_move: self.mpl_disconnect(self.cid_move)
        self.pressed = False

    def _on_press(self, event: MouseEvent):
        if event.inaxes != self.ax:
            return
        self.pressed = True
        app = self.parent().parent()
        app.canvas_clicked(self)  # notify main app
        if not self.zoom_enabled and self.on_drag:
            self.on_drag(event.xdata, event.ydata)
        self.prev_drag = (event.xdata, event.ydata)  # for panning

    def _on_move(self, event: MouseEvent):
        if event.inaxes != self.ax:
            self.draw_idle()
            return

        if not self.pressed:
            return
        else: # Mouse Drag
            if self.zoom_enabled:
                self._on_pan(event)
            else:
                if self.on_drag:
                    self.on_drag(event.xdata, event.ydata)

    def _on_release(self, event: MouseEvent):
        self.pressed = False
        self.prev_drag = None

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


class ViewerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Brain Viewer with Segmentation Tools')
        self.volume = None
        self.affine = None
        self.shape = None  # (X, Y, Z)
        self.pos = None

        self.t1_volume = None
        self.t1_affine = None
        self.t2_volume = None
        self.t2_affine = None
        self.flair_volume = None
        self.flair_affine = None
        self.adc_volume = None
        self.adc_affine = None
        self.current_modality = ""
        self.focused_canvas = None

        self.upsample_enabled = True
        self._init_upsample_factor = 1
        self.upsample_factor = float(self._init_upsample_factor)

        self._init_toolbar()
        self._init_layout()
        self._test_init() # initialize images

    def _init_toolbar(self):
        # ----------------------- TOP TOOL BAR -----------------------
        toolbar = QtWidgets.QToolBar("MainToolbar")
        self.addToolBar(QtCore.Qt.TopToolBarArea, toolbar)

        load_btn = QtWidgets.QAction("Load T1", self)
        load_btn.triggered.connect(self._load_t1_volume)
        toolbar.addAction(load_btn)

        toolbar.addSeparator()

        load_t2_btn = QtWidgets.QAction("Load T2", self)
        load_t2_btn.triggered.connect(self._load_t2_volume)
        toolbar.addAction(load_t2_btn)

        toolbar.addSeparator()

        load_flair_btn = QtWidgets.QAction("Load FLAIR", self)
        load_flair_btn.triggered.connect(self._load_flair_volume)
        toolbar.addAction(load_flair_btn)

        toolbar.addSeparator()

        load_adc_btn = QtWidgets.QAction("Load ADC", self)
        load_adc_btn.triggered.connect(self._load_adc_volume)
        toolbar.addAction(load_adc_btn)

        toolbar.addSeparator()

        load_dwi_btn = QtWidgets.QAction("Load DWI", self)
        # load_dwi_btn.triggered.connect()
        toolbar.addAction(load_dwi_btn)

        toolbar.addSeparator()
        

        self.modality_group = QtWidgets.QButtonGroup(self)
        self.t1_radio = QtWidgets.QRadioButton("T1")
        self.t2_radio = QtWidgets.QRadioButton("T2")
        self.flair_radio = QtWidgets.QRadioButton("FLAIR")
        self.adc_radio = QtWidgets.QRadioButton("ADC")
        self.dwi_radio = QtWidgets.QRadioButton("DWI")
        self.flair_radio.setChecked(False)
        self.t1_radio.setDisabled(True)
        self.t2_radio.setDisabled(True)
        self.flair_radio.setDisabled(True)
        self.adc_radio.setDisabled(True)
        self.dwi_radio.setDisabled(True)
        self.modality_group.addButton(self.t1_radio)
        self.modality_group.addButton(self.t2_radio)
        self.modality_group.addButton(self.flair_radio)
        self.modality_group.addButton(self.adc_radio)
        self.modality_group.addButton(self.dwi_radio)
        self.t1_radio.toggled.connect(self._change_modality)
        self.t2_radio.toggled.connect(self._change_modality)
        self.flair_radio.toggled.connect(self._change_modality)
        self.adc_radio.toggled.connect(self._change_modality)
        self.dwi_radio.toggled.connect(self._change_modality)
        toolbar.addWidget(self.t1_radio)
        toolbar.addWidget(self.t2_radio)
        toolbar.addWidget(self.flair_radio)
        toolbar.addWidget(self.adc_radio)
        toolbar.addWidget(self.dwi_radio)

        toolbar.addSeparator()

        self.zoom_checkbox = QtWidgets.QCheckBox("Zoom Mode")
        self.zoom_checkbox.setChecked(False)
        self.zoom_checkbox.stateChanged.connect(self._toggle_zoom_mode_checkbox)
        toolbar.addWidget(self.zoom_checkbox)

        toolbar.addSeparator()

        reset_btn = QtWidgets.QAction("Close all files", self)
        reset_btn.triggered.connect(self._reset_volumes)
        toolbar.addAction(reset_btn)

        # 2nd toolbar (filters and resolution) ----------------------------
        self.addToolBarBreak()            # force new row
        toolbar2 = QtWidgets.QToolBar("SecondaryToolbar")
        self.addToolBar(QtCore.Qt.TopToolBarArea, toolbar2)

        self.filter_checkbox = QtWidgets.QCheckBox("Apply Filter ")
        self.filter_checkbox.setChecked(False)
        self.filter_checkbox.stateChanged.connect(self._toggle_filter_checkbox)
        toolbar2.addWidget(self.filter_checkbox)

        toolbar2.addSeparator()

        self.upsample_checkbox = QtWidgets.QCheckBox("Resolution: ")
        self.upsample_checkbox.setChecked(True)
        self.upsample_checkbox.stateChanged.connect(self._toggle_upsample_checkbox)
        toolbar2.addWidget(self.upsample_checkbox)

        self.upsample_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.upsample_slider.setMinimum(1)
        self.upsample_slider.setMaximum(6)
        self.upsample_slider.setValue(int(self.upsample_factor))
        self.upsample_slider.setSingleStep(1)
        self.upsample_slider.setMaximumWidth(300)
        self.upsample_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.upsample_slider.setTickInterval(1)
        self.upsample_slider.valueChanged.connect(self._upsample_slider_change)
        self.upsample_slider_label = QtWidgets.QLabel(f" X {self.upsample_factor}  ")
        self.upsample_slider.valueChanged.connect(lambda value: self.upsample_slider_label.setText(f" X {self.upsample_factor}  "))
        toolbar2.addWidget(self.upsample_slider_label)
        toolbar2.addWidget(self.upsample_slider)

        toolbar2.addSeparator()

        self.axial_resolution_checkbox = QtWidgets.QCheckBox("Axial ")
        self.axial_resolution_checkbox.setChecked(True)
        self.axial_resolution_checkbox.stateChanged.connect(self._update_all)
        toolbar2.addWidget(self.axial_resolution_checkbox)

        self.coronal_resolution_checkbox = QtWidgets.QCheckBox("Coronal ")
        self.coronal_resolution_checkbox.setChecked(True)
        self.coronal_resolution_checkbox.stateChanged.connect(self._update_all)
        toolbar2.addWidget(self.coronal_resolution_checkbox)

        self.sagittal_resolution_checkbox = QtWidgets.QCheckBox("Sagittal ")
        self.sagittal_resolution_checkbox.setChecked(True)
        self.sagittal_resolution_checkbox.stateChanged.connect(self._update_all)
        toolbar2.addWidget(self.sagittal_resolution_checkbox)

    def _init_layout(self):
        icon_path = resource_path("resources/icons/app_icon.png")
        self.setWindowIcon(QtGui.QIcon(str(icon_path)))

        # ----------------------- MAIN WIDGET and layout -----------------------
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

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
        self.axial_canvas_2 = SliceCanvas(self, title="3D")

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
            message = f"position (x,y,z): {self.pos[0]}, {self.pos[1]}, {self.pos[2]}    "
            message += f"resolution: {self.shape[0] * self.upsample_factor:.0f}, {self.shape[1] * self.upsample_factor:.0f}, {self.shape[2] * self.upsample_factor:.0f}"
            self.status.showMessage(message)

    # ---------------- 2D slices ----------------
    def _get_normal_axial(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.axial_canvas_2.ax is not None:
            self.axial_canvas_2.ax.set_xlim((1 / self.upsample_factor) * np.array(self.axial_canvas.ax.get_xlim(), dtype=np.float64))
            self.axial_canvas_2.ax.set_ylim((1 / self.upsample_factor) * np.array(self.axial_canvas.ax.get_ylim(), dtype=np.float64))
        slice2d = self.volume[:, :, self.pos[2]]
        slice2d = np.fliplr(slice2d.T)
        return slice2d, None

    def _get_axial(self) -> Tuple[np.ndarray, np.ndarray]:
        slice2d = self.volume[:, :, self.pos[2]]
        slice2d = np.fliplr(slice2d.T)  # transpose for correct orientation

        if self.upsample_enabled and self.axial_resolution_checkbox.isChecked():
            slice2d = fast_mri_slice_upsample(slice2d, self.upsample_factor)

        return slice2d, None

    def _get_coronal(self) -> Tuple[np.ndarray, np.ndarray]:
        slice2d = self.volume[:, self.pos[1], :]
        slice2d = np.fliplr(slice2d.T)  # transpose for correct orientation
        
        if self.upsample_enabled and self.coronal_resolution_checkbox.isChecked():
            slice2d = fast_mri_slice_upsample(slice2d, self.upsample_factor)

        return slice2d, None

    def _get_sagittal(self) -> Tuple[np.ndarray, np.ndarray]:
        slice2d = self.volume[self.pos[0], :, :]
        slice2d = np.fliplr(slice2d.T)  # transpose for correct orientation
        
        if self.upsample_enabled and self.sagittal_resolution_checkbox.isChecked():
            slice2d = fast_mri_slice_upsample(slice2d, self.upsample_factor)

        return slice2d, None

    def _voxel_to_display(self, v):
        return (v + 0.5) * self._init_upsample_factor
    
    # ---------------- Update Everything -----------------------
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

    def _display_to_voxel(self, xpix: float, ypix: float) -> Tuple[int, int]:
        x = int((xpix - 0.5) / self._init_upsample_factor)
        y = int((ypix - 0.5) / self._init_upsample_factor)
        return x, y
    
    # ---------------- Drag callbacks ----------------
    def _on_axial_drag(self, xpix: float, ypix: float):
        x, y = self._display_to_voxel(xpix, ypix)
        self.pos[0] = np.clip(self.shape[0] - 1 - x, 0, self.shape[0] - 1)
        self.pos[1] = np.clip(y, 0, self.shape[1] - 1)
        self._update_all()

    def _on_coronal_drag(self, xpix: float, ypix: float):
        x, y = self._display_to_voxel(xpix, ypix)
        self.pos[0] = np.clip(self.shape[0] - 1 - x, 0, self.shape[0] - 1)
        self.pos[2] = np.clip(y, 0, self.shape[2] - 1)
        self._update_all()

    def _on_sagittal_drag(self, xpix: float, ypix: float):
        x, y = self._display_to_voxel(xpix, ypix)
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

        # ---------------- Key release ----------------
        elif event.type() == QtCore.QEvent.KeyRelease:

            modifiers = event.modifiers()

            # CTRL released → disable zoom
            if not (modifiers & QtCore.Qt.ControlModifier):
                self._toggle_zoom_mode(False)

        return super().eventFilter(obj, event)
    
    def focusOutEvent(self, event):
        self._toggle_brush_mode(False)
        self._toggle_zoom_mode(False)
        super().focusOutEvent(event)
        
    # ---------- Selecting Main Volume showed -----------------
    def _change_modality(self):
        if self.t1_radio.isChecked():
            if self.t1_volume is not None:
                self.current_modality = "T1"
                if self.filter_checkbox.isChecked():
                    self.volume = self._apply_volume_normalization(self.t1_volume, self.brainmask, pmin=1, pmax=99)
                else:
                    self.volume = self.t1_volume
                self.affine = self.t1_affine
        elif self.t2_radio.isChecked():
            if self.t2_volume is not None:
                self.current_modality = "T2"
                if self.filter_checkbox.isChecked():
                    self.volume = self._apply_volume_normalization(self.t2_volume, self.brainmask, pmin=1, pmax=99)
                else:
                    self.volume = self.t2_volume
                self.affine = self.t2_affine
        elif self.flair_radio.isChecked():
            if self.flair_radio is not None:
                self.current_modality = "FLAIR"
                if self.filter_checkbox.isChecked():
                    self.volume = self._apply_volume_normalization(self.flair_volume, self.brainmask, pmin=1, pmax=99)
                else:
                    self.volume = self.flair_volume
                self.affine = self.flair_affine
        else:
            if self.adc_volume is not None:
                self.current_modality = "ADC"
                if self.filter_checkbox.isChecked():
                    self.volume = self._apply_volume_normalization(self.adc_volume, self.brainmask, pmin=1, pmax=99)
                else:
                    self.volume = self.adc_volume
                self.affine = self.adc_affine

        self._update_all()

    def _toggle_filter_checkbox(self, state):
        enabled = (state == QtCore.Qt.Checked)
        if enabled:
            if self.current_modality == "FLAIR":
                self.volume = self._apply_volume_normalization(self.flair_volume, self.brainmask, pmin=1, pmax=99)
        else:
            if self.current_modality == "FLAIR":
                self.volume = self.flair_volume
        self._update_all()

    def _apply_volume_normalization(self, volume: np.ndarray, brainmask: np.ndarray = None, pmin=1, pmax=99) -> np.ndarray:
        volume = volume.copy()
        if brainmask is None:
            brainmask = np.ones_like(volume, dtype=bool)
        lo, hi = enhance.compute_volume_normalization(volume, brainmask, pmin, pmax)
        clipped = enhance.apply_volume_normalization(volume, lo, hi)
        mean = clipped.mean()
        std = clipped.std()

        # normalized = np.zeros_like(volume, dtype=np.float32)
        volume[brainmask == 1] = (volume[brainmask == 1] - mean) / std

        return volume

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
            self.upsample_slider.setEnabled(False)
            self.upsample_slider_label.setText(f" X {self.upsample_factor}  ")

        # self._update_canvases_figures()
        self._update_all()

    def _upsample_slider_change(self, value):
        self.upsample_factor = float(value)
        self._update_all()

    # ---------------- Load volumes ----------------
    def _load_t1_volume(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load NIfTI File", "", "NIfTI (*.nii *.nii.gz)"
        )
        if not path:
            return

        data, aff = load_volume(path, np.float32)
        self.t1_volume = data.astype(np.float32)
        self.t1_affine = aff
        self._load_new_volume(data, aff, modality="T1")
        self.t1_radio.setChecked(True)
        self.t1_radio.setEnabled(True)
        
        self._update_all()

    def _load_t2_volume(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load NIfTI File", "", "NIfTI (*.nii *.nii.gz)"
        )
        if not path:
            return

        data, aff = load_volume(path, np.float32)
        self.t2_volume = data.astype(np.float32)
        self.t2_affine = aff
        self._load_new_volume(data, aff, modality="T2")
        self.t2_radio.setChecked(True)
        self.t2_radio.setEnabled(True)
        
        self._update_all()

    def _load_flair_volume(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load NIfTI File", "", "NIfTI (*.nii *.nii.gz)"
        )
        if not path:
            return

        data, aff = load_volume(path, np.float32)
        self.flair_volume = data.astype(np.float32)
        self.flair_affine = aff
        self._load_new_volume(data, aff, modality="FLAIR")
        self.flair_radio.setChecked(True)
        self.flair_radio.setEnabled(True)
        
        self._update_all()

    def _load_adc_volume(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load NIfTI File", "", "NIfTI (*.nii *.nii.gz)"
        )
        if not path:
            return
        
        data, aff = load_volume(path, np.float32)
        self.adc_volume = data.astype(np.float32)
        self.adc_affine = aff
        self._load_new_volume(data, aff, modality="ADC")
        self.adc_radio.setChecked(True)
        self.adc_radio.setEnabled(True)
        
        self._update_all()
        
    def _load_new_volume(self, data: np.ndarray, aff: np.ndarray, modality: str = "T1"):
        self.current_modality = modality
        data = data.copy()
        aff = aff.copy()
        if self.filter_checkbox.isChecked():
            self.volume = self._apply_volume_normalization(data, self.brainmask, pmin=1, pmax=99)
        else:
            self.volume = data
        self.affine = aff

        self.shape = self.volume.shape[:3]
        self.pos = [s // 2 for s in self.shape]  # initial crosshair at center
        
        sx = np.linalg.norm(self.affine[:3,0])   # spacing along i axis (rows of data)
        sy = np.linalg.norm(self.affine[:3,1])   # spacing along j axis (cols of data)
        sz = np.linalg.norm(self.affine[:3,2])   # spacing along k axis (slice thickness)
        self.axial_canvas.aspect = sy / sx
        self.coronal_canvas.aspect = sz / sx
        self.sagittal_canvas.aspect = sz / sy
        self.axial_canvas_2.aspect = sy / sx

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

        self.upsample_factor = float(self._init_upsample_factor)
        self.upsample_slider.setValue(self._init_upsample_factor)

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
        self.shape = None
        self.pos = None

        self.focused_canvas = None
        
        self.axial_canvas.show_empty()
        self.coronal_canvas.show_empty()
        self.sagittal_canvas.show_empty()
        self.upsample_factor = float(self._init_upsample_factor)

    def _test_init(self):
        data, aff = load_volume("./test_nifti_files/abscess/FLAIR.nii.gz", np.float32)
        self.flair_volume = data.astype(np.float32)
        self.flair_affine = aff
        self._load_new_volume(data, aff, modality="FLAIR")
        self.flair_radio.setChecked(True)
        self.flair_radio.setEnabled(True)
        self.current_modality = "FLAIR"
        self._update_all()

        data, aff = load_volume("./test_nifti_files/abscess/ADC_registered_to_FLAIR.nii.gz", np.float32)
        self.adc_volume = data.astype(np.float32)
        self.adc_affine = aff
        self._load_new_volume(data, aff, modality="ADC")
        self.adc_radio.setChecked(True)
        self.adc_radio.setEnabled(True)
        self.current_modality = "ADC"
        self._update_all()


def main():
    app = QtWidgets.QApplication(sys.argv)
    viewer = ViewerApp()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
