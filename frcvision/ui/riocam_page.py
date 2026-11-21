"""'roboRIO camera' page - control a USB camera on the roboRIO without changing robot code."""

from __future__ import annotations

import time

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QWidget,
)

from ..camera import CameraConfig, VisionWorker, retire
from ..riocam import (
    CsCoreClient,
    RioCamera,
    RioCamError,
    apply_values,
    current_values,
    delete_profile,
    load_profiles,
    preset_values,
    rio_hosts,
    save_profile,
    scan,
)
from . import theme
from .limelight_page import Job
from .widgets import Card, VideoView, compact_combos, draw_label, hint, scroll_panel

RESOLUTIONS = ["Camera native", "640x480", "424x240", "320x240", "160x120"]
FPS = ["Camera native", "30", "20", "15", "10"]
QUALITY = ["Camera native", "80", "60", "40", "20"]

PRETTY = {
    "exposure_auto": "Exposure mode", "exposure_absolute": "Exposure", "exposure_time_absolute": "Exposure",
    "white_balance_temperature_auto": "Auto white balance", "white_balance_temperature": "White balance (K)",
    "white_balance_automatic": "Auto white balance", "backlight_compensation": "Backlight compensation",
    "power_line_frequency": "Power line (anti-flicker)", "focus_auto": "Autofocus", "focus_absolute": "Focus",
    "exposure_dynamic_framerate": "Dynamic frame rate", "connect_verbose": "Verbose log",
}
# controls shown first (most useful for driving / vision)
ORDER = ["exposure_auto", "exposure_absolute", "exposure_time_absolute", "gain", "brightness", "contrast",
         "saturation", "sharpness", "white_balance_temperature_auto", "white_balance_automatic",
         "white_balance_temperature", "backlight_compensation", "power_line_frequency", "focus_auto",
         "focus_absolute", "zoom_absolute"]


class ControlRow(QWidget):
    """One camera control (bool / int / enum / string) that reports changes via on_change(name, value)."""

    def __init__(self, ctrl, on_change, parent=None):
        super().__init__(parent)
        self.ctrl = ctrl
        self.on_change = on_change
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.widget = None
        if ctrl.kind == "bool":
            w = QCheckBox("on")
            w.setChecked(ctrl.int_value() != 0)
            w.toggled.connect(lambda v: self.on_change(ctrl.name, 1 if v else 0))
            lay.addWidget(w)
            lay.addStretch(1)
        elif ctrl.kind == "enum" and ctrl.menu:
            w = QComboBox()
            for k, label in sorted(ctrl.menu.items()):
                w.addItem(f"{label}  ({k})", k)
            idx = w.findData(ctrl.int_value())
            if idx >= 0:
                w.setCurrentIndex(idx)
            w.currentIndexChanged.connect(lambda _i: self.on_change(ctrl.name, w.currentData()))
            lay.addWidget(w, 1)
        elif ctrl.kind in ("int", "enum"):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(ctrl.min, max(ctrl.max, ctrl.min))
            s.setSingleStep(ctrl.step)
            s.setValue(ctrl.int_value())
            sp = QSpinBox()
            sp.setRange(ctrl.min, max(ctrl.max, ctrl.min))
            sp.setSingleStep(ctrl.step)
            sp.setValue(ctrl.int_value())
            sp.setFixedWidth(72)
            s.valueChanged.connect(sp.setValue)
            sp.valueChanged.connect(s.setValue)
            sp.valueChanged.connect(lambda v: self.on_change(ctrl.name, v))
            lay.addWidget(s, 1)
            lay.addWidget(sp)
            w = sp
        else:
            w = QLineEdit(ctrl.value)
            w.editingFinished.connect(lambda: self.on_change(ctrl.name, w.text()))
            lay.addWidget(w, 1)
        self.widget = w
        tip = f"{ctrl.name}  [{ctrl.kind}]  range {ctrl.min}..{ctrl.max} step {ctrl.step}  default {ctrl.default}"
        self.setToolTip(tip)

    def set_value(self, value: str):
        w = self.widget
        w.blockSignals(True)
        try:
            if isinstance(w, QCheckBox):
                w.setChecked(int(float(value)) != 0)
            elif isinstance(w, QComboBox):
                i = w.findData(int(float(value)))
                if i >= 0:
                    w.setCurrentIndex(i)
            elif isinstance(w, QSpinBox):
                w.setValue(int(float(value)))
                slider = self.findChild(QSlider)
                if slider:
                    slider.blockSignals(True)
                    slider.setValue(int(float(value)))
                    slider.blockSignals(False)
            elif isinstance(w, QLineEdit):
                w.setText(value)
        except ValueError:
            pass
        finally:
            w.blockSignals(False)


class RioCamPage(QWidget):
    mode = "idle"

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.cameras: list[RioCamera] = []
        self.cam: RioCamera | None = None
        self.client: CsCoreClient | None = None
        self.stream: VisionWorker | None = None
        self.rows: dict[str, ControlRow] = {}
        self._pending: dict = {}
        self._jobs: list[Job] = []
        self._last_sent = ""
        self._send_timer = QTimer(self)
        self._send_timer.setSingleShot(True)
        self._send_timer.setInterval(180)  # debounce slider drags
        self._send_timer.timeout.connect(self._flush)
        self._keep_timer = QTimer(self)
        self._keep_timer.setInterval(5000)
        self._keep_timer.timeout.connect(self._keep_alive)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.view = VideoView()
        self.view.placeholder = "roboRIO camera - find it on the right"
        self.view.overlay = self._overlay
        root.addWidget(self.view, 1)
        panel, lay = scroll_panel(410)
        root.addWidget(panel)

        # ---------------------------------------------------------- connect
        c = Card("roboRIO camera", "A USB camera on the roboRIO, served by WPILib CameraServer "
                                   "(startAutomaticCapture). Settings go straight to the camera over HTTP - "
                                   "no robot-code changes, the robot program does not see them.")
        self.host = QComboBox()
        self.host.setEditable(True)
        self.team = QSpinBox()
        self.team.setRange(0, 25999)
        self.team.setSpecialValueText("-")
        self.team.valueChanged.connect(self._fill_hosts)
        c.add_row("Team number", self.team)
        c.add_row("roboRIO address", self.host)
        self.find_btn = QPushButton("Find cameras")
        self.find_btn.setObjectName("Primary")
        self.find_btn.clicked.connect(self.find_cameras)
        c.add(self.find_btn)
        self.cam_combo = QComboBox()
        self.cam_combo.currentIndexChanged.connect(self._select_camera)
        c.add_row("Camera", self.cam_combo)
        self.status = hint("Scans ports 1181-1190 on the roboRIO.")
        c.add(self.status)
        lay.addWidget(c)

        # ---------------------------------------------------------- stream
        s = Card("Stream", "Changes only what this app receives (saves field bandwidth, 4 Mbit/s limit). "
                           "The camera's own capture mode stays as the robot code set it.")
        self.show_stream = QCheckBox("Show live video")
        self.show_stream.setChecked(True)
        self.show_stream.toggled.connect(lambda _v: self._restart_stream())
        s.add(self.show_stream)
        self.res_combo = QComboBox()
        self.res_combo.addItems(RESOLUTIONS)
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(FPS)
        self.q_combo = QComboBox()
        self.q_combo.addItems(QUALITY)
        for w in (self.res_combo, self.fps_combo, self.q_combo):
            w.currentIndexChanged.connect(lambda _i: self._restart_stream())
        s.add_row("Resolution", self.res_combo)
        s.add_row("Frame rate", self.fps_combo)
        s.add_row("JPEG quality", self.q_combo)
        use = QPushButton("Use as the app camera (AprilTags / calibration)")
        use.clicked.connect(self._use_as_camera)
        s.add(use)
        lay.addWidget(s)

        # ---------------------------------------------------------- presets
        p = Card("Quick presets", "Built from the controls this camera actually has.")
        row = QHBoxLayout()
        for key, label, tip in (("auto", "Auto", "Auto exposure + auto white balance, defaults"),
                                ("fast", "Fast motion", "Short exposure (less blur while driving), more gain"),
                                ("dim", "Low light", "Longer exposure for a dark venue, autofocus off")):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: self.apply_preset(k))
            row.addWidget(b)
        p.add_layout(row)
        lay.addWidget(p)

        # ---------------------------------------------------------- controls
        self.ctrl_card = Card("Camera settings")
        self.show_raw = QCheckBox("Show raw device values")
        self.show_raw.toggled.connect(lambda _v: self._build_controls())
        self.ctrl_card.header_right.addWidget(self.show_raw)
        self.ctrl_host = QWidget()
        self.ctrl_form = None
        self.ctrl_card.add(self.ctrl_host)
        r = QHBoxLayout()
        ref = QPushButton("Refresh")
        ref.clicked.connect(self.refresh)
        dft = QPushButton("Reset to defaults")
        dft.clicked.connect(self.reset_defaults)
        r.addWidget(ref)
        r.addWidget(dft)
        self.ctrl_card.add_layout(r)
        self.ctrl_msg = hint("")
        self.ctrl_card.add(self.ctrl_msg)
        lay.addWidget(self.ctrl_card)

        # ---------------------------------------------------------- profiles
        pf = Card("Profiles", "Camera settings live on the camera until it loses power. Save a profile and "
                              "re-apply it after a reboot - or let the app do it automatically.")
        self.profile_combo = QComboBox()
        pf.add(self.profile_combo)
        r2 = QHBoxLayout()
        sv = QPushButton("Save current…")
        sv.clicked.connect(self._save_profile)
        ap = QPushButton("Apply")
        ap.setObjectName("Primary")
        ap.clicked.connect(lambda: self.apply_profile(self.profile_combo.currentData()))
        dl = QPushButton("Delete")
        dl.setObjectName("Danger")
        dl.clicked.connect(self._delete_profile)
        r2.addWidget(sv)
        r2.addWidget(ap, 1)
        r2.addWidget(dl)
        pf.add_layout(r2)
        self.keep = QCheckBox("Keep applied (auto re-apply)")
        self.keep.toggled.connect(self._toggle_keep)
        pf.add(self.keep)
        pf.add(hint("Every 5 s: if the camera came back with different settings (robot reboot), the selected "
                    "profile is re-applied. Works while this app is open - also on the driver station during "
                    "matches (ports 1180-1190 are open on the field network)."))
        self.profile_msg = hint("")
        pf.add(self.profile_msg)
        lay.addWidget(pf)
        lay.addStretch(1)

        self._fill_hosts()
        self._reload_profiles()
        compact_combos(self)

    # ================================================================ connect
    def _fill_hosts(self, *_):
        cur = self.host.currentText()
        self.host.blockSignals(True)
        self.host.clear()
        self.host.addItems(rio_hosts(self.team.value()))
        if cur and cur not in rio_hosts(self.team.value()):
            self.host.addItem(cur)
            self.host.setCurrentText(cur)
        self.host.blockSignals(False)

    def _run(self, fn, done):
        job = Job(fn, self)
        self._jobs.append(job)

        def finished(res, err):
            self._jobs.remove(job)
            done(res, err)

        job.done.connect(finished)
        job.start()

    def find_cameras(self):
        host = self.host.currentText().strip()
        if not host:
            return
        self.find_btn.setEnabled(False)
        self.status.setText(f"Scanning {host} ports 1181-1190…")

        def done(res, err):
            self.find_btn.setEnabled(True)
            if err:
                self.status.setText(f"<span style='color:{theme.BAD}'>{err}</span>")
                return
            self.cameras = res or []
            self.cam_combo.blockSignals(True)
            self.cam_combo.clear()
            for cam in self.cameras:
                self.cam_combo.addItem(cam.label, cam.port)
            self.cam_combo.blockSignals(False)
            if not self.cameras:
                self.status.setText(
                    f"<span style='color:{theme.WARN}'>No camera server found on {host}.</span> Check that the "
                    "robot is on and the robot code calls CameraServer.startAutomaticCapture() "
                    "(the only line the robot needs).")
                self._set_camera(None)
                return
            self.status.setText(f"<span style='color:{theme.ACCENT}'>&#9679; Found {len(self.cameras)} camera"
                                f"{'s' if len(self.cameras) != 1 else ''} on {host}</span>")
            self.cam_combo.setCurrentIndex(0)
            self._select_camera(0)

        self._run(lambda: scan(host), done)

    def _select_camera(self, idx):
        if 0 <= idx < len(self.cameras):
            self._set_camera(self.cameras[idx])

    def _set_camera(self, cam: RioCamera | None):
        self.cam = cam
        self.client = CsCoreClient(cam.host, cam.port) if cam else None
        self._build_controls()
        self._restart_stream()
        self.ctx.status(f"roboRIO camera: {cam.host}:{cam.port}" if cam else "roboRIO camera: none")

    # ================================================================ stream
    def _stream_url(self):
        if not self.cam:
            return None
        res = None if self.res_combo.currentIndex() == 0 else self.res_combo.currentText()
        fps = None if self.fps_combo.currentIndex() == 0 else int(self.fps_combo.currentText())
        q = None if self.q_combo.currentIndex() == 0 else int(self.q_combo.currentText())
        return self.cam.stream_url(res, fps, q)

    def _restart_stream(self):
        self._stop_stream()
        url = self._stream_url()
        if not url or not self.show_stream.isChecked():
            self.view.clear("Stream hidden" if url else "roboRIO camera - find it on the right")
            return
        self.view.clear(f"Opening {url}…")
        w = VisionWorker(CameraConfig(name="roboRIO camera", url=url), self)
        w.packet_ready.connect(self._on_frame)
        w.failed.connect(lambda m: self.view.clear(m))
        self.stream = w
        w.start()

    def _stop_stream(self):
        if self.stream is not None:
            s, self.stream = self.stream, None
            retire(s)

    def _on_frame(self):
        if self.stream is None:
            return
        pkt = self.stream.take_latest()
        if pkt is not None and self.isVisible():
            self.view.set_image(pkt.image, pkt.fps)

    def _overlay(self, p, view):
        if self.cam is None:
            return
        r = view.image_rect()
        fps = view.data if isinstance(view.data, float) else 0.0
        w, h = view.image_size
        draw_label(p, QPointF(r.x() + 8, r.y() + 28), f"roboRIO {self.cam.host}:{self.cam.port}  |  "
                                                      f"{w}x{h}  {fps:.0f} fps", fg=theme.ACCENT,
                   align_center=False, size=10)

    def _use_as_camera(self):
        url = self._stream_url()
        if not url:
            self.ctx.status("Find a roboRIO camera first.")
            return
        self.show_stream.setChecked(False)
        cam = self.ctx.pages["camera"]
        cam.set_stream(url)
        cfg = cam.current_config()
        cfg.name = f"roboRIO {self.cam.host}:{self.cam.port}"
        self.ctx.connect_camera(cfg)
        self.ctx.goto_page("apriltag")

    # ================================================================ controls
    def _build_controls(self):
        old = self.ctrl_host.layout()
        if old is not None:
            while old.count():
                item = old.takeAt(0)
                if item.widget() is not None:
                    item.widget().deleteLater()
            QWidget().setLayout(old)  # hand the empty layout to a temporary widget so it is freed
        self.rows.clear()
        form = QFormLayout(self.ctrl_host)
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(6)
        if not self.cam:
            form.addRow(hint("No camera selected."))
            return
        ctrls = [c for c in self.cam.controls if self.show_raw.isChecked() or not c.raw]
        ctrls.sort(key=lambda c: (ORDER.index(c.name) if c.name in ORDER else len(ORDER), c.name))
        for ctrl in ctrls:
            row = ControlRow(ctrl, self._queue)
            label = QLabel(PRETTY.get(ctrl.name, ctrl.name.replace("_", " ").capitalize()))
            label.setObjectName("Muted")
            label.setToolTip(ctrl.name)
            form.addRow(label, row)
            self.rows[ctrl.name] = row
        self.ctrl_msg.setText(f"{len(ctrls)} controls. Changes are sent as you move them.")

    def _queue(self, name, value):
        self._pending[name] = value
        self._send_timer.start()

    def _flush(self):
        if not self.client or not self._pending:
            return
        values, self._pending = dict(self._pending), {}
        client, cam = self.client, self.cam

        def work():
            apply_values(client, cam, values)
            return client.fetch()

        def done(res, err):
            if err:
                self.ctrl_msg.setText(f"<span style='color:{theme.BAD}'>{err}</span>")
                return
            self._sync(res)
            self.ctrl_msg.setText("Applied: " + ", ".join(f"{k}={v}" for k, v in values.items()))

        self._run(work, done)

    def _sync(self, fresh: RioCamera):
        """Take the camera's real values (some settings change others, e.g. auto exposure)."""
        if not self.cam or fresh.port != self.cam.port:
            return
        self.cam.controls, self.cam.modes, self.cam.config = fresh.controls, fresh.modes, fresh.config
        for c in fresh.controls:
            row = self.rows.get(c.name)
            if row is not None and c.name not in self._pending:
                row.set_value(c.value)

    def refresh(self):
        if not self.client:
            return
        client = self.client
        self._run(client.fetch, lambda res, err: self.ctrl_msg.setText(str(err)) if err else self._sync(res))

    def reset_defaults(self):
        if not self.cam:
            return
        vals = {c.name: (c.default if c.kind == "string" else int(float(c.default or 0)))
                for c in self.cam.controls if not c.raw and c.default != ""}
        self._apply(vals, "defaults")

    def apply_preset(self, key):
        if not self.cam:
            self.ctx.status("Find a roboRIO camera first.")
            return
        vals = preset_values(self.cam, key)
        if not vals:
            self.ctrl_msg.setText("This camera has none of the controls that preset uses.")
            return
        self._apply(vals, f"preset '{key}'")

    def _apply(self, vals: dict, what: str, quiet=False):
        client, cam = self.client, self.cam

        def work():
            skipped = apply_values(client, cam, vals)
            return client.fetch(), skipped

        def done(res, err):
            if err:
                self.ctrl_msg.setText(f"<span style='color:{theme.BAD}'>{err}</span>")
                return
            fresh, skipped = res
            self._sync(fresh)
            msg = f"Applied {what}: {len(vals) - len(skipped)} setting(s)"
            if skipped:
                msg += f" (camera has no {', '.join(skipped)})"
            if not quiet:
                self.ctrl_msg.setText(msg)
            self.ctx.status(msg)

        self._run(work, done)

    # ================================================================ profiles
    def _reload_profiles(self, select=None):
        self.profile_combo.clear()
        for name, vals in sorted(load_profiles().items()):
            self.profile_combo.addItem(f"{name}  ({len(vals)} settings)", name)
        if select:
            i = self.profile_combo.findData(select)
            if i >= 0:
                self.profile_combo.setCurrentIndex(i)

    def _save_profile(self):
        if not self.cam:
            self.ctx.status("Find a roboRIO camera first.")
            return
        name, ok = QInputDialog.getText(self, "Camera profile", "Profile name:", text="Driver camera")
        if ok and name.strip():
            save_profile(name.strip(), current_values(self.cam))
            self._reload_profiles(name.strip())
            self.profile_msg.setText(f"Saved '{name.strip()}'.")

    def _delete_profile(self):
        name = self.profile_combo.currentData()
        if name and QMessageBox.question(self, "Delete profile", f"Delete '{name}'?") == QMessageBox.StandardButton.Yes:
            delete_profile(name)
            self._reload_profiles()

    def apply_profile(self, name, quiet=False):
        vals = load_profiles().get(name or "")
        if not vals or not self.cam:
            if not quiet:
                self.profile_msg.setText("Select a camera and a profile first.")
            return
        self._apply(vals, f"profile '{name}'", quiet)
        if not quiet:
            self.profile_msg.setText(f"Applied '{name}'.")

    def _toggle_keep(self, on):
        if on:
            self._keep_timer.start()
            self._keep_alive()
        else:
            self._keep_timer.stop()

    def _keep_alive(self):
        """Re-find the camera if needed and re-apply the profile when its settings drifted."""
        name = self.profile_combo.currentData()
        vals = load_profiles().get(name or "")
        host = self.host.currentText().strip()
        if not vals or not host or self._jobs:
            return
        port = self.cam.port if self.cam else None

        def work():
            cams = [CsCoreClient(host, port).fetch()] if port else scan(host)
            if not cams:
                return None, []
            cam = cams[0]
            cur = current_values(cam)
            drift = {k: v for k, v in vals.items() if k in cur and str(cur[k]) != str(v)}
            if drift:
                apply_values(CsCoreClient(cam.host, cam.port), cam, vals)
            return cam, drift

        def done(res, err):
            if err or not res or res[0] is None:
                self.profile_msg.setText(f"Keep applied: waiting for the camera on {host}…")
                return
            cam, drift = res
            if self.cam is None or self.cam.port != cam.port:
                self.cameras = [cam]
                self.cam_combo.blockSignals(True)
                self.cam_combo.clear()
                self.cam_combo.addItem(cam.label, cam.port)
                self.cam_combo.blockSignals(False)
                self._set_camera(cam)
            stamp = time.strftime("%H:%M:%S")
            if drift:
                self.profile_msg.setText(f"{stamp} re-applied '{name}' ({len(drift)} setting(s) had changed).")
                self.refresh()
            else:
                self.profile_msg.setText(f"{stamp} '{name}' is active on the camera.")

        self._run(work, done)

    # ================================================================ misc
    def on_packet(self, _pkt):
        pass

    def shutdown(self):
        self._keep_timer.stop()
        self._stop_stream()
        for j in list(self._jobs):
            j.wait(3000)
