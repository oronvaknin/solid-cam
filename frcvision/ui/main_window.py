"""Main window: header, sidebar navigation, pages and the camera worker."""

from __future__ import annotations

import time
from dataclasses import asdict

import cv2
from PySide6.QtCore import QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..apriltag import AprilTagPipeline
from ..calibration import BoardConfig, CameraCalibration, default_calibration_path
from ..camera import CameraConfig, VisionWorker, retire, wait_retired
from ..state import AppState
from . import theme
from .apriltag_page import AprilTagPage
from .calibration_page import CalibrationPage
from .camera_page import CameraPage
from .icons import make_icons
from .limelight_page import LimelightPage
from .riocam_page import RioCamPage
from .widgets import Chip

NAV = [
    ("VISION", None, None),
    ("apriltag", "AprilTags", "apriltag"),
    ("calibration", "Calibration", "calibration"),
    ("DEVICES", None, None),
    ("camera", "Camera", "camera"),
    ("limelight", "Limelight", "limelight"),
    ("riocam", "roboRIO camera", "riocam"),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"FRC Vision Lab {__version__}")
        self.resize(1560, 940)
        self.state = AppState()
        self.worker: VisionWorker | None = None
        self.camera_config = CameraConfig()
        self.apriltag_pipeline = AprilTagPipeline(self.state.apriltag)
        self._chip_t = 0.0
        self.settings = QSettings("FRCVisionLab", "FRCVisionLab")

        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        v.addLayout(body, 1)

        self.stack = QStackedWidget()
        self.pages = {
            "camera": CameraPage(self),
            "apriltag": AprilTagPage(self),
            "calibration": CalibrationPage(self),
            "limelight": LimelightPage(self),
            "riocam": RioCamPage(self),
        }
        for p in self.pages.values():
            self.stack.addWidget(p)
        body.addWidget(self._build_sidebar())
        body.addWidget(self.stack, 1)

        self.setStatusBar(QStatusBar())
        self.state.calibration_changed.connect(self._update_calib_chip)
        self.pages["limelight"].status_changed.connect(lambda t, c: self.chip_ll.set_state(t, c))

        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._space)
        for i, key in enumerate(("apriltag", "calibration", "camera", "limelight", "riocam"), 1):
            QShortcut(QKeySequence(f"Ctrl+{i}"), self, activated=lambda k=key: self.goto_page(k))

        self._restore_settings()
        self.goto_page("apriltag")
        self._update_calib_chip()
        QTimer.singleShot(200, lambda: self.connect_camera(self.camera_config))

    # ================================================================ chrome
    def _build_header(self):
        h = QFrame()
        h.setObjectName("Header")
        h.setFixedHeight(58)
        lay = QHBoxLayout(h)
        lay.setContentsMargins(18, 0, 14, 0)
        mark = QLabel()
        mark.setObjectName("LogoMark")
        mark.setFixedSize(30, 30)
        lay.addWidget(mark)
        lay.addSpacing(6)
        col = QVBoxLayout()
        col.setSpacing(0)
        logo = QLabel("FRC VISION LAB")
        logo.setObjectName("Logo")
        sub = QLabel("AprilTags  •  3D field  •  Calibration  •  Limelight")
        sub.setObjectName("LogoSub")
        col.addStretch(1)
        col.addWidget(logo)
        col.addWidget(sub)
        col.addStretch(1)
        lay.addLayout(col)
        lay.addStretch(1)
        self.chip_cam = Chip()
        self.chip_res = Chip()
        self.chip_fps = Chip()
        self.chip_lat = Chip()
        self.chip_cal = Chip()
        self.chip_ll = Chip()
        for c in (self.chip_cam, self.chip_res, self.chip_fps, self.chip_lat, self.chip_cal, self.chip_ll):
            lay.addWidget(c)
            lay.addSpacing(4)
        self.chip_cam.set_state("No camera", theme.BAD)
        self.chip_res.set_state("-")
        self.chip_fps.set_state("0 fps")
        self.chip_lat.set_state("0 ms")
        self.chip_ll.set_state("LL offline", theme.MUTED)
        self.chip_ll.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chip_ll.mousePressEvent = lambda _e: self.goto_page("limelight")
        return h

    def _build_sidebar(self):
        s = QFrame()
        s.setObjectName("Sidebar")
        s.setFixedWidth(186)
        lay = QVBoxLayout(s)
        lay.setContentsMargins(0, 10, 0, 12)
        lay.setSpacing(2)
        self.nav = QButtonGroup(self)
        self.nav.setExclusive(True)
        self.nav_buttons = {}
        icons = make_icons(theme.MUTED, theme.ACCENT)
        for key, text, icon in NAV:
            if text is None:
                lab = QLabel(key)
                lab.setObjectName("NavSection")
                lay.addSpacing(8)
                lay.addWidget(lab)
                continue
            b = QPushButton("  " + text)
            b.setObjectName("NavButton")
            b.setCheckable(True)
            b.setIcon(icons[icon])
            b.setIconSize(QSize(20, 20))
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=key: self.goto_page(k))
            self.nav.addButton(b)
            self.nav_buttons[key] = b
            lay.addWidget(b)
        lay.addStretch(1)
        tip = QLabel("Ctrl+1..5  switch pages\nSpace  capture (calibration)")
        tip.setObjectName("Hint")
        tip.setContentsMargins(16, 0, 8, 6)
        lay.addWidget(tip)
        ver = QLabel(f"v{__version__}")
        ver.setObjectName("Hint")
        ver.setContentsMargins(16, 0, 0, 0)
        lay.addWidget(ver)
        return s

    # ================================================================== API
    def status(self, text: str, ms: int = 6000):
        self.statusBar().showMessage(text, ms)

    def goto_page(self, key: str):
        page = self.pages[key]
        self.stack.setCurrentWidget(page)
        self.nav_buttons[key].setChecked(True)
        if self.worker is not None:
            self.worker.mode = page.mode

    def current_page(self):
        return self.stack.currentWidget()

    def set_camera_property(self, prop: int, value: float):
        if self.worker is not None:
            self.worker.set_property(prop, value)

    def open_driver_settings(self):
        if self.worker is None:
            self.status("Connect a camera first.")
            return
        if self.camera_config.backend != "DirectShow" or self.camera_config.url:
            self.status("Driver settings dialog needs a USB camera on the DirectShow backend.")
        self.worker.open_driver_settings()

    def use_stream_as_camera(self, url: str, name: str):
        cam = self.pages["camera"]
        cam.set_stream(url)
        cfg = cam.current_config()
        cfg.name = name
        self.connect_camera(cfg)
        self.goto_page("calibration")
        self.status(f"Camera source: {url} - calibrate, then upload from the Limelight page.")

    # ---------------------------------------------------------------- camera
    def connect_camera(self, cfg: CameraConfig):
        self.disconnect_camera(quiet=True)
        self.camera_config = cfg
        self.state.camera_name = cfg.name
        self.chip_cam.set_state(f"Connecting {cfg.name}…", theme.WARN)
        for key in ("camera", "apriltag", "calibration"):
            self.pages[key].view.clear("Connecting…")
        w = VisionWorker(cfg, self)
        w.processors = {
            "raw": self._proc_raw,
            "apriltag": self._proc_apriltag,
            "charuco": self.pages["calibration"].process,
        }
        w.mode = self.current_page().mode
        w.packet_ready.connect(self._on_packet)
        w.opened.connect(self._on_opened)
        w.failed.connect(self._on_failed)
        self.worker = w
        w.start()
        self.pages["camera"].apply_config(cfg)

    def disconnect_camera(self, quiet=False):
        if self.worker is not None:
            w, self.worker = self.worker, None
            retire(w)
        if not quiet:
            self.chip_cam.set_state("Disconnected", theme.MUTED)
            for key in ("camera", "apriltag", "calibration"):
                self.pages[key].view.clear("Camera disconnected")

    def _on_opened(self, info: dict):
        w, h = info["width"], info["height"]
        self.state.frame_size = (w, h)
        self.chip_cam.set_state(self.camera_config.name, theme.ACCENT)
        self.chip_res.set_state(f"{w}x{h}", theme.TEXT)
        cam_page = self.pages["camera"]
        cam_page.on_opened(info)
        if not self.camera_config.url:
            cam_page._exposure_mode(cam_page.auto_exp.isChecked())
        if not self.camera_config.url and (w, h) != (self.camera_config.width, self.camera_config.height):
            self.status(f"Camera chose {w}x{h} (requested {self.camera_config.width}x{self.camera_config.height}).")
        else:
            self.status(f"Connected: {self.camera_config.name} {w}x{h} via {info.get('backend')}")
        self._auto_load_calibration(w, h)

    def _on_failed(self, msg: str):
        self.chip_cam.set_state("Camera error", theme.BAD)
        self.status(msg, 15000)
        for key in ("camera", "apriltag", "calibration"):
            self.pages[key].view.clear(msg + "\nChoose a camera on the Camera page (Ctrl+3).")
        if self.worker is not None and not self.worker.isRunning():
            self.worker = None

    def _auto_load_calibration(self, w, h):
        cur = self.state.calibration
        if cur is not None and (cur.width, cur.height) == (w, h):
            return
        path = default_calibration_path(self.camera_config.name, w, h)
        if path.exists():
            try:
                self.state.set_calibration(CameraCalibration.load(path), str(path))
                self.status(f"Loaded calibration {path.name}")
            except Exception as exc:  # corrupt file
                self.status(f"Could not load {path.name}: {exc}")

    def _on_packet(self):
        if self.worker is None:
            return
        pkt = self.worker.take_latest()
        if pkt is None:
            return
        page = self.current_page()
        if pkt.mode == page.mode:
            page.on_packet(pkt)
        now = time.monotonic()
        if now - self._chip_t > 0.3:
            self._chip_t = now
            fps_col = theme.ACCENT if pkt.fps >= 20 else theme.WARN
            self.chip_fps.set_state(f"{pkt.fps:.0f} fps", fps_col)
            self.chip_lat.set_state(f"{pkt.latency_ms:.1f} ms pipeline", theme.TEXT)

    # ------------------------------------------------------------ processors
    def _proc_raw(self, frame):
        small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return frame, {"brightness": float(g.mean()),
                       "sharpness": float(cv2.Laplacian(g, cv2.CV_64F).var())}

    def _proc_apriltag(self, frame):
        h, w = frame.shape[:2]
        calib = self.state.intrinsics(w, h)
        res = self.apriltag_pipeline.process(frame, calib, self.state.layout, self.state.T_robot_camera)
        return frame, res

    # ------------------------------------------------------------------ misc
    def _update_calib_chip(self):
        c = self.state.calibration
        if c is not None and self.state.use_calibration:
            self.chip_cal.set_state(f"Calibrated  {c.rms:.2f}px", theme.ACCENT)
        else:
            self.chip_cal.set_state("Uncalibrated", theme.WARN)

    def _space(self):
        if self.current_page() is self.pages["calibration"]:
            self.pages["calibration"].capture()

    # -------------------------------------------------------------- settings
    def _restore_settings(self):
        s = self.settings
        cam = s.value("camera")
        if isinstance(cam, dict):
            try:
                self.camera_config = _coerce(CameraConfig, cam)
            except (TypeError, ValueError):
                pass
        self.pages["camera"].apply_config(self.camera_config)
        board = s.value("board")
        if isinstance(board, dict):
            try:
                self.pages["calibration"].set_board_config(_coerce(BoardConfig, board))
            except (TypeError, ValueError, KeyError):
                pass
        ap = self.pages["apriltag"]
        mount = s.value("mount")
        if isinstance(mount, dict):
            ap.set_mount_values(mount)
        layout = s.value("layout")
        if isinstance(layout, str) and layout:
            ap.select_layout(layout)
        try:
            ap.robot_len.setValue(float(s.value("robot_len", 0.8)))
            ap.robot_wid.setValue(float(s.value("robot_wid", 0.8)))
            ap.alliance.setCurrentText(str(s.value("alliance", "Blue")))
        except (TypeError, ValueError):
            pass
        ll = self.pages["limelight"]
        host = s.value("ll_host")
        if isinstance(host, str) and host:
            ll.host.setCurrentText(host)
        try:
            ll.team.setValue(int(s.value("team", 0)))
        except (TypeError, ValueError):
            pass
        hosts = s.value("ll_tune_hosts")
        if isinstance(hosts, str):
            hosts = [hosts]
        for h in hosts or []:
            ll._add_host(str(h))
        rio = self.pages["riocam"]
        try:
            rio.team.setValue(int(s.value("rio_team", ll.team.value())))
        except (TypeError, ValueError):
            pass
        rio_host = s.value("rio_host")
        if isinstance(rio_host, str) and rio_host:
            rio.host.setCurrentText(rio_host)
        prof = s.value("rio_profile")
        if isinstance(prof, str) and prof:
            rio._reload_profiles(prof)
        geo = s.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)

    def _save_settings(self):
        s = self.settings
        s.setValue("camera", asdict(self.camera_config))
        s.setValue("board", asdict(self.pages["calibration"].board_cfg))
        ap = self.pages["apriltag"]
        s.setValue("mount", ap.mount_values())
        s.setValue("robot_len", ap.robot_len.value())
        s.setValue("robot_wid", ap.robot_wid.value())
        s.setValue("alliance", ap.alliance.currentText())
        if self.state.layout is not None:
            s.setValue("layout", self.state.layout.path)
        s.setValue("ll_host", self.pages["limelight"].host.currentText())
        s.setValue("team", self.pages["limelight"].team.value())
        ll = self.pages["limelight"]
        s.setValue("ll_tune_hosts", [ll.ll_list.item(i).text() for i in range(ll.ll_list.count())])
        rio = self.pages["riocam"]
        s.setValue("rio_team", rio.team.value())
        s.setValue("rio_host", rio.host.currentText())
        s.setValue("rio_profile", rio.profile_combo.currentData() or "")
        s.setValue("geometry", self.saveGeometry())

    def closeEvent(self, ev):
        self._save_settings()
        self.pages["apriltag"].shutdown()
        self.pages["limelight"].shutdown()
        self.pages["riocam"].shutdown()
        self.disconnect_camera(quiet=True)
        wait_retired()
        super().closeEvent(ev)


def _coerce(cls, stored: dict):
    """Rebuild a dataclass from QSettings data (the registry may return strings)."""
    base = asdict(cls())
    out = {}
    for k, default in base.items():
        if k not in stored:
            out[k] = default
            continue
        v = stored[k]
        if isinstance(default, bool):
            out[k] = v in (True, "true", "True", "1", 1)
        elif isinstance(default, int):
            out[k] = int(float(v))
        elif isinstance(default, float):
            out[k] = float(v)
        else:
            out[k] = str(v)
    return cls(**out)
