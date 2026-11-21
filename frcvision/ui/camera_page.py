"""'Camera' page - pick the USB camera, resolution and exposure (Limelight 'Input' tab)."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QWidget,
)

from ..camera import BACKENDS, FPS_OPTIONS, RESOLUTIONS, CameraConfig, list_camera_names
from . import theme
from .widgets import Card, VideoView, compact_combos, hint, scroll_panel, value_label

SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "snapshots"


class LabeledSlider(QWidget):
    def __init__(self, lo, hi, value, fmt="{:.0f}", parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(lo), int(hi))
        self.slider.setValue(int(value))
        self.label = QLabel(fmt.format(value))
        self.label.setObjectName("Value")
        self.label.setMinimumWidth(42)
        self.label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.fmt = fmt
        self.slider.valueChanged.connect(lambda v: self.label.setText(self.fmt.format(v)))
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.label)


class CameraPage(QWidget):
    mode = "raw"

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._last_ui = 0.0
        self._last_frame = None

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.view = VideoView()
        root.addWidget(self.view, 1)
        panel, lay = scroll_panel()
        root.addWidget(panel)

        # ---------------------------------------------------------- device
        dev = Card("Camera device", "Cameras connected to this computer (USB webcam, Limelight USB, etc.).")
        row = QHBoxLayout()
        self.cam_combo = QComboBox()
        self.cam_combo.setMinimumWidth(180)
        refresh = QPushButton("↻")
        refresh.setToolTip("Re-scan cameras")
        refresh.setFixedWidth(34)
        refresh.clicked.connect(self.refresh_cameras)
        row.addWidget(self.cam_combo, 1)
        row.addWidget(refresh)
        dev.add_layout(row)
        self.url_edit = QLineEdit("http://limelight.local:5800")
        self.url_edit.setPlaceholderText("http://10.TE.AM.11:5800  or  rtsp://...")
        self.url_edit.setToolTip("MJPEG / RTSP stream, e.g. Limelight :5800, PhotonVision :1182/stream.mjpg")
        dev.add(self.url_edit)
        self.cam_combo.currentIndexChanged.connect(self._source_changed)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(list(BACKENDS))
        dev.add_row("Backend", self.backend_combo)
        self.res_combo = QComboBox()
        for w, h in RESOLUTIONS:
            self.res_combo.addItem(f"{w} x {h}", (w, h))
        dev.add_row("Resolution", self.res_combo)
        self.fps_combo = QComboBox()
        for f in FPS_OPTIONS:
            self.fps_combo.addItem(f"{f} fps", f)
        dev.add_row("Frame rate", self.fps_combo)
        self.mjpg_check = QCheckBox("MJPG stream (needed for high res / fps on most webcams)")
        self.mjpg_check.setChecked(True)
        dev.add(self.mjpg_check)
        btns = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setObjectName("Primary")
        self.connect_btn.clicked.connect(self._connect)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(lambda: self.ctx.disconnect_camera())
        btns.addWidget(self.connect_btn, 1)
        btns.addWidget(self.disconnect_btn)
        dev.add_layout(btns)
        lay.addWidget(dev)

        # ---------------------------------------------------------- image
        img = Card("Image", "For AprilTags use a short exposure (less motion blur), then raise gain/brightness.")
        self.auto_exp = QCheckBox("Auto exposure")
        self.auto_exp.setChecked(True)
        self.auto_exp.toggled.connect(self._exposure_mode)
        img.add(self.auto_exp)
        self.exposure = LabeledSlider(-13, -1, -6)
        self.exposure.slider.valueChanged.connect(
            lambda v: self.ctx.set_camera_property(cv2.CAP_PROP_EXPOSURE, float(v)))
        img.add_row("Exposure (log2 s)", self.exposure)
        self.gain = LabeledSlider(0, 255, 0)
        self.gain.slider.valueChanged.connect(lambda v: self.ctx.set_camera_property(cv2.CAP_PROP_GAIN, float(v)))
        img.add_row("Gain", self.gain)
        self.brightness = LabeledSlider(0, 255, 128)
        self.brightness.slider.valueChanged.connect(
            lambda v: self.ctx.set_camera_property(cv2.CAP_PROP_BRIGHTNESS, float(v)))
        img.add_row("Brightness", self.brightness)
        drv = QPushButton("Open camera driver settings…")
        drv.setToolTip("Windows DirectShow property dialog (full exposure / focus / white-balance control)")
        drv.clicked.connect(lambda: self.ctx.open_driver_settings())
        img.add(drv)
        img.add(hint("Tip: turn OFF autofocus in the driver settings for stable AprilTag poses and calibration."))
        self._exposure_mode(True)
        lay.addWidget(img)

        # ---------------------------------------------------------- stream info
        info = Card("Stream")
        self.v_res = value_label()
        self.v_fps = value_label()
        self.v_backend = value_label()
        self.v_bright = value_label()
        self.v_sharp = value_label()
        info.add_row("Resolution", self.v_res)
        info.add_row("Measured FPS", self.v_fps)
        info.add_row("Backend", self.v_backend)
        info.add_row("Mean brightness", self.v_bright)
        info.add_row("Sharpness", self.v_sharp)
        snap = QPushButton("Save snapshot (PNG)")
        snap.clicked.connect(self._snapshot)
        info.add(snap)
        lay.addWidget(info)
        lay.addStretch(1)

        self.refresh_cameras()

    # ------------------------------------------------------------------ API
    def refresh_cameras(self):
        current = self.cam_combo.currentData()
        self.cam_combo.clear()
        names = list_camera_names()
        for i, n in enumerate(names):
            self.cam_combo.addItem(f"{i}: {n}", (i, n))
        for i in range(len(names), max(len(names), 1) + 3):
            self.cam_combo.addItem(f"{i}: (camera index {i})", (i, f"Camera {i}"))
        self.cam_combo.addItem("Network stream (Limelight / MJPEG URL)", (-1, "Network stream"))
        if current is not None:
            for i in range(self.cam_combo.count()):
                if tuple(self.cam_combo.itemData(i)) == tuple(current):
                    self.cam_combo.setCurrentIndex(i)
                    break
        self._source_changed()
        compact_combos(self)

    def set_stream(self, url: str):
        for i in range(self.cam_combo.count()):
            if self.cam_combo.itemData(i)[0] == -1:
                self.cam_combo.setCurrentIndex(i)
                break
        self.url_edit.setText(url)
        self._source_changed()

    def _source_changed(self, *_):
        data = self.cam_combo.currentData()
        is_url = bool(data) and data[0] == -1
        self.url_edit.setVisible(is_url)
        for w in (self.backend_combo, self.res_combo, self.fps_combo, self.mjpg_check):
            w.setEnabled(not is_url)

    def apply_config(self, cfg: CameraConfig):
        for combo, val in ((self.res_combo, (cfg.width, cfg.height)), (self.fps_combo, cfg.fps)):
            for i in range(combo.count()):  # findData() can't match Python tuples reliably
                if tuple(combo.itemData(i)) == tuple(val) if isinstance(val, tuple) else combo.itemData(i) == val:
                    combo.setCurrentIndex(i)
                    break
        self.backend_combo.setCurrentText(cfg.backend)
        self.mjpg_check.setChecked(cfg.mjpg)
        want = -1 if cfg.url else cfg.index
        if cfg.url:
            self.url_edit.setText(cfg.url)
        for i in range(self.cam_combo.count()):
            if self.cam_combo.itemData(i)[0] == want:
                self.cam_combo.setCurrentIndex(i)
                break
        self._source_changed()

    def current_config(self) -> CameraConfig:
        idx, name = self.cam_combo.currentData() or (0, "Camera 0")
        w, h = self.res_combo.currentData()
        url = ""
        if idx == -1:
            url = self.url_edit.text().strip()
            name = "stream " + url.split("//")[-1].split("/")[0]
            idx = 0
        return CameraConfig(index=idx, name=name, backend=self.backend_combo.currentText(),
                            width=w, height=h, fps=self.fps_combo.currentData(),
                            mjpg=self.mjpg_check.isChecked(), url=url)

    def on_opened(self, info: dict):
        self.v_res.setText(f"{info['width']} x {info['height']}")
        self.v_backend.setText(str(info.get("backend", "")))
        for slider, key in ((self.gain, "gain"), (self.brightness, "brightness")):
            v = info.get(key)
            if v is not None and v >= 0:
                slider.slider.blockSignals(True)
                slider.slider.setValue(int(v))
                slider.label.setText(slider.fmt.format(int(v)))
                slider.slider.blockSignals(False)

    def on_packet(self, pkt):
        self._last_frame = pkt.frame
        self.view.set_image(pkt.image, None)
        now = time.monotonic()
        if now - self._last_ui > 0.25:
            self._last_ui = now
            self.v_fps.setText(f"{pkt.fps:.1f}")
            if isinstance(pkt.result, dict):
                b = pkt.result.get("brightness", 0)
                color = theme.WARN if b < 40 or b > 215 else theme.TEXT
                self.v_bright.setText(f"<span style='color:{color}'>{b:.0f} / 255</span>")
                self.v_sharp.setText(f"{pkt.result.get('sharpness', 0):.0f}")

    # -------------------------------------------------------------- private
    def _connect(self):
        self.ctx.connect_camera(self.current_config())

    def _exposure_mode(self, auto: bool):
        self.exposure.setEnabled(not auto)
        # DirectShow/V4L convention: 0.75 = auto, 0.25 = manual
        self.ctx.set_camera_property(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if auto else 0.25)
        if not auto:
            self.ctx.set_camera_property(cv2.CAP_PROP_EXPOSURE, float(self.exposure.slider.value()))

    def _snapshot(self):
        if self._last_frame is None:
            return
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        default = SNAPSHOT_DIR / time.strftime("snapshot_%Y%m%d_%H%M%S.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save snapshot", str(default), "PNG (*.png)")
        if path:
            cv2.imwrite(path, self._last_frame)
            self.ctx.status(f"Snapshot saved: {path}")
