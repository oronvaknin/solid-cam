"""'AprilTags' page - detect every FRC field tag, per-tag 3D data and field (bot) pose."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..apriltag import REFINE_METHODS, AprilTagResult, pose_summary, robot_to_camera_T
from ..field_layout import FieldLayout, bundled_layouts
from . import theme
from .field3d import Field3DView, RobotView
from .field_view import FieldView
from .widgets import Card, VideoView, compact_combos, draw_label, hint, value_label

REC_DIR = Path(__file__).resolve().parents[2] / "recordings"


def parse_id_filter(text: str):
    ids = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            ids.update(range(int(a), int(b) + 1))
        else:
            ids.add(int(part))
    return frozenset(ids) if ids else None


def _tab_scroll() -> tuple[QScrollArea, QVBoxLayout]:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(12, 12, 12, 12)
    lay.setSpacing(12)
    sa.setWidget(w)
    return sa, lay


def _spin(lo, hi, val, step=0.01, dec=3, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    return s


class AprilTagPage(QWidget):
    mode = "apriltag"

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.state = ctx.state
        self._last_ui = 0.0
        self._last_result: AprilTagResult | None = None

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        split = QSplitter(Qt.Orientation.Vertical)
        self.view = VideoView()
        self.view.overlay = self._paint_overlay
        self.field = FieldView()
        self.field3d = Field3DView()
        fieldbox = QWidget()
        fl = QVBoxLayout(fieldbox)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(0)
        seg = QWidget()
        seg.setObjectName("SegBar")
        sl = QHBoxLayout(seg)
        sl.setContentsMargins(10, 6, 10, 6)
        sl.setSpacing(4)
        self.field_stack = QStackedWidget()
        self.field_stack.addWidget(self.field3d)
        self.field_stack.addWidget(self.field)
        for i, name in enumerate(("3D field", "2D map")):
            b = QPushButton(name)
            b.setObjectName("Seg")
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.setChecked(i == 0)
            b.clicked.connect(lambda _=False, i=i: self.field_stack.setCurrentIndex(i))
            sl.addWidget(b)
        sl.addStretch(1)
        fl.addWidget(seg)
        fl.addWidget(self.field_stack, 1)
        split.addWidget(self.view)
        split.addWidget(fieldbox)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([520, 400])
        root.addWidget(split, 1)
        self._recorder = None

        panel = QWidget()
        panel.setObjectName("Panel")
        panel.setMinimumWidth(400)
        panel.setMaximumWidth(500)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)
        tabs = QTabWidget()
        pl.addWidget(tabs)
        root.addWidget(panel)

        tabs.addTab(self._build_targets_tab(), "Targets")
        tabs.addTab(self._build_pipeline_tab(), "Pipeline")
        tabs.addTab(self._build_robot_tab(), "Robot / 3D")

        self.state.calibration_changed.connect(self._refresh_intrinsics)
        self._load_default_layout()
        self._refresh_intrinsics()
        compact_combos(self)

    # =================================================================== tabs
    def _build_pipeline_tab(self):
        sa, lay = _tab_scroll()
        s = self.state.apriltag

        fm = Card("Field map", "Official FRC tag positions (WPILib, blue-alliance origin).")
        self.layout_combo = QComboBox()
        for p in bundled_layouts():
            self.layout_combo.addItem(p.stem, str(p))
        self.layout_combo.currentIndexChanged.connect(
            lambda _i: self._load_layout(self.layout_combo.currentData()))
        fm.add(self.layout_combo)
        load = QPushButton("Load field map file (.json / .fmap)…")
        load.clicked.connect(self._browse_layout)
        fm.add(load)
        self.layout_info = hint("")
        fm.add(self.layout_info)
        lay.addWidget(fm)

        det = Card("Detector", "Family tag36h11 (FRC 2024+).")
        self.size_spin = _spin(10, 1000, s.tag_size_m * 1000, 0.1, 1, " mm")
        self.size_spin.valueChanged.connect(lambda v: setattr(s, "tag_size_m", v / 1000.0))
        det.add_row("Tag size", self.size_spin)
        self.downscale = QComboBox()
        for d in (1.0, 1.5, 2.0, 3.0, 4.0):
            self.downscale.addItem(f"{d:g}x", d)
        self.downscale.currentIndexChanged.connect(
            lambda _i: setattr(s, "downscale", self.downscale.currentData()))
        det.add_row("Detector downscale", self.downscale)
        self.refine = QComboBox()
        for label, key in REFINE_METHODS.items():
            self.refine.addItem(label, key)
        self.refine.currentIndexChanged.connect(lambda _i: setattr(s, "refine", self.refine.currentData()))
        det.add_row("Corner refine", self.refine)
        self.id_filter = QLineEdit()
        self.id_filter.setPlaceholderText("all IDs   (e.g. 3,4,19-20)")
        self.id_filter.editingFinished.connect(self._apply_filter)
        det.add_row("ID filter", self.id_filter)
        self.multitag = QCheckBox("MegaTag-style multi-tag pose (all visible tags together)")
        self.multitag.setChecked(s.multitag)
        self.multitag.toggled.connect(lambda v: setattr(s, "multitag", v))
        det.add(self.multitag)
        self.amb = _spin(0.0, 1.0, s.max_single_tag_ambiguity, 0.05, 2)
        self.amb.valueChanged.connect(lambda v: setattr(s, "max_single_tag_ambiguity", v))
        det.add_row("Max single-tag ambiguity", self.amb)
        self.cube = QCheckBox("Draw 3D cube on tags")
        self.cube.setChecked(s.draw_cube)
        self.cube.toggled.connect(lambda v: setattr(s, "draw_cube", v))
        det.add(self.cube)
        lay.addWidget(det)

        intr = Card("Camera intrinsics")
        self.intr_status = QLabel()
        self.intr_status.setWordWrap(True)
        intr.add(self.intr_status)
        self.use_calib = QCheckBox("Use camera calibration")
        self.use_calib.setChecked(True)
        self.use_calib.toggled.connect(self._toggle_calib)
        intr.add(self.use_calib)
        self.hfov = _spin(30, 170, self.state.hfov_guess, 1, 1, " deg")
        self.hfov.valueChanged.connect(self._set_hfov)
        intr.add_row("Assumed HFOV (uncalibrated)", self.hfov)
        go = QPushButton("Calibrate this camera →")
        go.clicked.connect(lambda: self.ctx.goto_page("calibration"))
        intr.add(go)
        lay.addWidget(intr)
        lay.addStretch(1)
        return sa

    def _build_targets_tab(self):
        sa, lay = _tab_scroll()
        top = Card("Primary target", "Largest tag in view (Limelight tx / ty / ta).")
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(4)
        self.big_id = QLabel("-")
        self.big_id.setObjectName("Big")
        grid.addWidget(QLabel("ID"), 0, 0)
        grid.addWidget(self.big_id, 1, 0, 2, 1)
        self.pv = {}
        keys = [("tx", "tx (deg)"), ("ty", "ty (deg)"), ("ta", "ta (%)"),
                ("dist", "Distance (m)"), ("fwd", "Forward (m)"), ("left", "Left (m)"),
                ("up", "Up (m)"), ("skew", "Tag yaw (deg)"), ("amb", "Ambiguity")]
        for n, (k, label) in enumerate(keys):
            r, c = divmod(n, 3)
            lab = QLabel(label)
            lab.setObjectName("Hint")
            v = value_label()
            self.pv[k] = v
            grid.addWidget(lab, r * 2, c + 1)
            grid.addWidget(v, r * 2 + 1, c + 1)
        top.add_layout(grid)
        lay.addWidget(top)

        tc = Card("All visible tags")
        self.count_label = QLabel("0 tags")
        self.count_label.setObjectName("Muted")
        tc.header_right.addWidget(self.count_label)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["ID", "tx°", "ty°", "ta %", "Dist m", "Yaw°", "Amb"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setMinimumHeight(260)
        tc.add(self.table)
        tc.add(hint("Green ID = tag is in the field map (used for robot pose). "
                    "Ambiguity < 0.2 is trustworthy for a single tag."))
        lay.addWidget(tc)

        acc = Card("Accuracy check", "Measure the real distance from the camera lens to the primary tag with a "
                                     "tape measure and enter it - the error tells you how good the calibration is.")
        self.true_dist = _spin(0.0, 20.0, 0.0, 0.01, 3, " m")
        self.true_dist.setSpecialValueText("enter measured distance")
        acc.add_row("Measured distance", self.true_dist)
        self.acc_label = QLabel("-")
        self.acc_label.setObjectName("Value")
        acc.add(self.acc_label)
        lay.addWidget(acc)

        bp = Card("Bot pose (field)")
        self.bp_label = QLabel("No field-map tags in view")
        self.bp_label.setObjectName("Value")
        self.bp_label.setWordWrap(True)
        bp.add(self.bp_label)
        lay.addWidget(bp)
        lay.addStretch(1)
        return sa

    def _build_robot_tab(self):
        sa, lay = _tab_scroll()
        out = Card("Robot pose - wpiblue", "Same convention as Limelight botpose_wpiblue: "
                                             "origin at the blue alliance corner, metres / degrees.")
        self.rv = {}
        grid = QGridLayout()
        for n, k in enumerate(["x", "y", "z", "roll", "pitch", "yaw"]):
            r, c = divmod(n, 3)
            lab = QLabel(k)
            lab.setObjectName("Hint")
            v = value_label()
            v.setStyleSheet("font-size: 13pt;")
            self.rv[k] = v
            grid.addWidget(lab, r * 2, c)
            grid.addWidget(v, r * 2 + 1, c)
        out.add_layout(grid)
        self.rv_meta = hint("")
        out.add(self.rv_meta)
        copy = QPushButton("Copy botpose array")
        copy.clicked.connect(self._copy_botpose)
        out.add(copy)
        lay.addWidget(out)

        cam = Card("Camera pose in robot space",
                   "Where the camera is mounted relative to the robot centre (floor level). "
                   "Leave at 0 to get the camera's own pose.")
        self.mount = {}
        specs = [("forward", "Forward", -2, 2, " m"), ("left", "Left", -2, 2, " m"), ("up", "Up", -1, 3, " m"),
                 ("roll", "Roll", -180, 180, " deg"), ("pitch", "Pitch (tilt up +)", -90, 90, " deg"),
                 ("yaw", "Yaw (CCW +)", -180, 180, " deg")]
        for key, label, lo, hi, suf in specs:
            sp = _spin(lo, hi, 0.0, 0.01 if "m" in suf else 1.0, 3 if "m" in suf else 1, suf)
            sp.valueChanged.connect(self._update_mount)
            self.mount[key] = sp
            cam.add_row(label, sp)
        lay.addWidget(cam)

        fv = Card("Robot on the field view")
        self.robot_len = _spin(0.3, 2.0, 0.8, 0.01, 2, " m")
        self.robot_wid = _spin(0.3, 2.0, 0.8, 0.01, 2, " m")
        self.robot_len.valueChanged.connect(self._robot_size)
        self.robot_wid.valueChanged.connect(self._robot_size)
        fv.add_row("Length (with bumpers)", self.robot_len)
        fv.add_row("Width (with bumpers)", self.robot_wid)
        self.alliance = QComboBox()
        self.alliance.addItems(["Blue", "Red"])
        self.alliance.currentTextChanged.connect(lambda t: setattr(self.field3d, "alliance", t.lower()))
        fv.add_row("Bumper colour", self.alliance)
        clr = QPushButton("Clear pose trail")
        clr.clicked.connect(self._clear_trails)
        fv.add(clr)
        lay.addWidget(fv)

        rec = Card("Record session (CSV)", "Logs every frame: time, tags, tx/ty/ta/distance, robot pose. "
                                           "Open it in Excel / AdvantageScope-style tools to analyse jitter.")
        self.rec_btn = QPushButton("●  Start recording")
        self.rec_btn.setCheckable(True)
        self.rec_btn.toggled.connect(self._toggle_record)
        rec.add(self.rec_btn)
        self.rec_label = hint("")
        rec.add(self.rec_label)
        lay.addWidget(rec)
        lay.addStretch(1)
        return sa

    # ============================================================ extras
    def _robot_size(self, *_):
        size = (self.robot_len.value(), self.robot_wid.value())
        self.field.robot_size = size
        self.field3d.robot_size = size
        self.field.update()
        self.field3d.update()

    def _clear_trails(self):
        self.field.clear_trail()
        self.field3d.clear_trails()

    def _toggle_record(self, on: bool):
        if on:
            REC_DIR.mkdir(exist_ok=True)
            path = REC_DIR / time.strftime("session_%Y%m%d_%H%M%S.csv")
            f = open(path, "w", newline="", encoding="utf-8")
            w = csv.writer(f)
            w.writerow(["time_s", "tag_ids", "primary_id", "tx", "ty", "ta", "dist_m", "ambiguity",
                        "pose_x", "pose_y", "pose_z", "roll", "pitch", "yaw", "method", "pose_tags", "reproj_px"])
            self._recorder = (f, w, time.monotonic(), path, [0])
            self.rec_btn.setText("■  Stop recording")
            self.rec_btn.setStyleSheet(f"color: {theme.BAD};")
            self.rec_label.setText(f"Recording to {path.parent.name}/{path.name}")
        else:
            if self._recorder:
                f, _w, _t0, path, n = self._recorder
                f.close()
                self.rec_label.setText(f"Saved {n[0]} rows: {path}")
                self.ctx.status(f"Recording saved: {path}")
            self._recorder = None
            self.rec_btn.setText("●  Start recording")
            self.rec_btn.setStyleSheet("")

    def _record(self, res: AprilTagResult):
        f, w, t0, _path, n = self._recorder
        prim = max(res.tags, key=lambda t: t.ta) if res.tags else None
        row = [f"{time.monotonic() - t0:.3f}", " ".join(str(t.id) for t in sorted(res.tags, key=lambda t: t.id))]
        if prim:
            row += [prim.id, f"{prim.tx:.3f}", f"{prim.ty:.3f}", f"{prim.ta:.3f}",
                    "" if prim.distance is None else f"{prim.distance:.4f}",
                    "" if prim.ambiguity is None else f"{prim.ambiguity:.3f}"]
        else:
            row += [""] * 6
        if res.pose:
            row += [f"{v:.4f}" for v in pose_summary(res.pose.T_field_robot)]
            row += [res.pose.method, " ".join(map(str, res.pose.tag_ids)), f"{res.pose.reproj_err:.3f}"]
        else:
            row += [""] * 9
        w.writerow(row)
        n[0] += 1

    def shutdown(self):
        if self._recorder:
            self.rec_btn.setChecked(False)

    # ============================================================ settings io
    def mount_values(self) -> dict:
        return {k: s.value() for k, s in self.mount.items()}

    def set_mount_values(self, values: dict):
        for k, v in (values or {}).items():
            if k in self.mount:
                self.mount[k].setValue(float(v))
        self._update_mount()

    def _update_mount(self):
        m = self.mount_values()
        self.state.T_robot_camera = robot_to_camera_T(m["forward"], m["left"], m["up"],
                                                      m["roll"], m["pitch"], m["yaw"])

    def _apply_filter(self):
        try:
            self.state.apriltag.id_filter = parse_id_filter(self.id_filter.text())
            self.id_filter.setStyleSheet("")
        except ValueError:
            self.id_filter.setStyleSheet(f"border-color: {theme.BAD};")

    def _toggle_calib(self, v):
        self.state.use_calibration = v
        self._refresh_intrinsics()

    def _set_hfov(self, v):
        self.state.hfov_guess = v
        self._refresh_intrinsics()

    # ================================================================ layout
    def _load_default_layout(self):
        if self.layout_combo.count():
            self._load_layout(self.layout_combo.currentData())

    def select_layout(self, path: str):
        i = self.layout_combo.findData(path)
        if i >= 0:
            self.layout_combo.setCurrentIndex(i)
        elif Path(path).exists():
            self._load_layout(path)

    def _browse_layout(self):
        path, _ = QFileDialog.getOpenFileName(self, "Field map", "", "Field maps (*.json *.fmap)")
        if path:
            self._load_layout(path)
            if self.layout_combo.findData(path) < 0:
                self.layout_combo.blockSignals(True)
                self.layout_combo.addItem(Path(path).stem, path)
                self.layout_combo.setCurrentIndex(self.layout_combo.count() - 1)
                self.layout_combo.blockSignals(False)

    def _load_layout(self, path):
        if not path:
            return
        try:
            layout = FieldLayout.load(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self.ctx.status(f"Could not load field map: {exc}")
            return
        self.state.set_layout(layout)
        self.size_spin.blockSignals(True)
        self.size_spin.setValue(layout.tag_size * 1000)
        self.size_spin.blockSignals(False)
        self.field.set_layout(layout)
        self.field3d.set_layout(layout)
        ids = sorted(layout.tags)
        self.layout_info.setText(f"{len(ids)} tags (IDs {ids[0]}-{ids[-1]}) | field "
                                 f"{layout.length:.3f} x {layout.width:.3f} m | tag {layout.tag_size * 1000:.1f} mm")
        self.ctx.status(f"Field map loaded: {layout.name}")

    def _refresh_intrinsics(self):
        c = self.state.calibration
        if c is not None and self.state.use_calibration:
            h, _v, _d = c.fov_deg()
            self.field.hfov_deg = h
            self.intr_status.setText(
                f"<span style='color:{theme.ACCENT}'>&#9679; Calibrated</span> &nbsp; {c.width}x{c.height}, "
                f"RMS {c.rms:.3f} px, HFOV {h:.1f}&deg;<br><span style='color:{theme.MUTED}'>"
                f"{Path(self.state.calibration_path).name or 'in memory'}</span>")
        else:
            self.field.hfov_deg = self.state.hfov_guess
            self.intr_status.setText(
                f"<span style='color:{theme.WARN}'>&#9679; Not calibrated</span> &nbsp; using an estimated "
                f"{self.state.hfov_guess:.0f}&deg; HFOV pinhole model. Distances and poses will be approximate.")

    # ================================================================ frames
    def on_packet(self, pkt):
        res = pkt.result if isinstance(pkt.result, AprilTagResult) else None
        self._last_result = res
        self.view.set_image(pkt.image, res)
        if isinstance(pkt.result, Exception):
            self.ctx.status(f"Pipeline error: {pkt.result}")
        if res is None:
            return
        ids = [t.id for t in res.tags]
        self.field.set_state(res.pose, ids)
        self.field3d.set_visible(ids)
        if res.pose is not None:
            hfov, vfov, _ = self.state.intrinsics(*res.image_size).fov_deg()
            self.field3d.set_robot("app", RobotView(res.pose.T_field_robot, res.pose.T_field_camera,
                                                    res.pose.tag_ids, theme.ACCENT, "This camera", hfov, vfov))
        else:
            self.field3d.set_robot("app", None)
        if self._recorder:
            self._record(res)
        now = time.monotonic()
        if now - self._last_ui < 0.1:
            return
        self._last_ui = now
        self._update_tables(res)

    def _update_tables(self, res: AprilTagResult):
        tags = sorted(res.tags, key=lambda t: t.id)
        self.count_label.setText(f"{len(tags)} tag{'s' if len(tags) != 1 else ''}")
        self.table.setRowCount(len(tags))
        for r, t in enumerate(tags):
            vals = [str(t.id), f"{t.tx:.2f}", f"{t.ty:.2f}", f"{t.ta:.2f}",
                    "-" if t.distance is None else f"{t.distance:.2f}",
                    "-" if t.tag_yaw is None else f"{t.tag_yaw:.1f}",
                    "-" if t.ambiguity is None else f"{t.ambiguity:.2f}"]
            for c, v in enumerate(vals):
                item = self.table.item(r, c)
                if item is None:
                    item = QTableWidgetItem()
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.table.setItem(r, c, item)
                item.setText(v)
                if c == 0:
                    item.setForeground(QColor(theme.ACCENT if t.in_layout else theme.WARN))
                elif c == 6 and t.ambiguity is not None:
                    item.setForeground(QColor(theme.BAD if t.ambiguity > 0.2 else theme.TEXT))
                else:
                    item.setForeground(QColor(theme.TEXT))

        prim = max(res.tags, key=lambda t: t.ta) if res.tags else None
        self.big_id.setText(str(prim.id) if prim else "-")
        self.big_id.setStyleSheet(f"color: {theme.ACCENT if prim else theme.MUTED};")

        def f(v, d=2):
            return "-" if v is None else f"{v:.{d}f}"

        self.pv["tx"].setText(f(prim.tx) if prim else "-")
        self.pv["ty"].setText(f(prim.ty) if prim else "-")
        self.pv["ta"].setText(f(prim.ta) if prim else "-")
        self.pv["dist"].setText(f(prim.distance) if prim else "-")
        xyz = prim.cam_xyz if prim and prim.cam_xyz else (None, None, None)
        self.pv["fwd"].setText(f(xyz[0]))
        self.pv["left"].setText(f(xyz[1]))
        self.pv["up"].setText(f(xyz[2]))
        self.pv["skew"].setText(f(prim.tag_yaw, 1) if prim else "-")
        self.pv["amb"].setText(f(prim.ambiguity) if prim else "-")
        truth = self.true_dist.value()
        if truth > 0 and prim is not None and prim.distance:
            err = prim.distance - truth
            pct = 100 * err / truth
            col = theme.ACCENT if abs(pct) < 2 else (theme.WARN if abs(pct) < 5 else theme.BAD)
            self.acc_label.setText(f"<span style='color:{col}'>tag {prim.id}: app {prim.distance:.3f} m  |  "
                                   f"error {err * 100:+.1f} cm ({pct:+.1f}%)</span>")
        else:
            self.acc_label.setText("-")

        pose = res.pose
        if pose is None:
            msg = "No field-map tags in view" if not any(t.in_layout for t in res.tags) else \
                "Pose rejected (single tag with high ambiguity)"
            self.bp_label.setText(f"<span style='color:{theme.MUTED}'>{msg}</span>")
            for v in self.rv.values():
                v.setText("-")
            self.rv_meta.setText(msg)
            return
        x, y, z, roll, pitch, yaw = pose_summary(pose.T_field_robot)
        self.bp_label.setText(
            f"<span style='color:{theme.ACCENT}; font-size:12pt'>x {x:.2f} &nbsp; y {y:.2f} &nbsp; "
            f"yaw {yaw:.1f}&deg;</span><br>"
            f"{pose.method} | tags {pose.tag_ids} | avg dist {pose.avg_dist:.2f} m | "
            f"reproj {pose.reproj_err:.2f} px")
        for k, v in zip(["x", "y", "z", "roll", "pitch", "yaw"], [x, y, z, roll, pitch, yaw]):
            self.rv[k].setText(f"{v:.3f}" if k in "xyz" else f"{v:.2f}°")
        amb = f" | ambiguity {pose.ambiguity:.2f}" if pose.ambiguity is not None else ""
        self.rv_meta.setText(f"{pose.method} | tags {pose.tag_ids} | avg dist {pose.avg_dist:.2f} m | "
                             f"reprojection {pose.reproj_err:.2f} px{amb}")

    def _copy_botpose(self):
        res = self._last_result
        if not res or not res.pose:
            self.ctx.status("No pose to copy")
            return
        x, y, z, roll, pitch, yaw = pose_summary(res.pose.T_field_robot)
        arr = [round(v, 4) for v in (x, y, z, roll, pitch, yaw)] + [
            len(res.pose.tag_ids), round(res.pose.avg_dist, 3)]
        QGuiApplication.clipboard().setText(json.dumps(arr))
        self.ctx.status(f"Copied botpose_wpiblue [x,y,z,roll,pitch,yaw,tagCount,avgDist]: {arr}")

    # ================================================================ overlay
    def _paint_overlay(self, p, view):
        res: AprilTagResult = view.data
        if res is None:
            return
        for t in res.tags:
            color = QColor(theme.ACCENT if t.in_layout else theme.WARN)
            if t.cube is not None:
                pts = [view.map(*pt) for pt in t.cube]
                edge = QColor(color)
                edge.setAlpha(170)
                p.setPen(QPen(edge, 1.5))
                for a, b in [(4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
                    p.drawLine(pts[a], pts[b])
            poly = QPolygonF([view.map(*c) for c in t.corners])
            fill = QColor(color)
            fill.setAlpha(45)
            p.setBrush(fill)
            p.setPen(QPen(color, 2.5))
            p.drawPolygon(poly)
            p.setBrush(QColor(theme.INFO))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(view.map(*t.corners[0]), 3.5, 3.5)  # top-left corner marker
            p.setBrush(Qt.BrushStyle.NoBrush)
            top = min(c[1] for c in t.corners)
            if t.cube is not None:
                top = min(top, min(pt[1] for pt in t.cube))
            label = f"ID {t.id}"
            if t.distance is not None:
                label += f"  {t.distance:.2f} m"
            draw_label(p, QPointF(view.map(t.center[0], top).x(), view.map(0, top).y() - 4), label,
                       fg=color.name())
        if res.pose is not None:
            x, y, _z, _r, _p, yaw = pose_summary(res.pose.T_field_robot)
            draw_label(p, QPointF(view.image_rect().x() + 8, view.image_rect().y() + 28),
                       f"BOTPOSE  x {x:.2f}  y {y:.2f}  yaw {yaw:.1f}°  ({res.pose.method})",
                       align_center=False, size=10)
        if not res.calibrated:
            r = view.image_rect()
            draw_label(p, QPointF(r.right() - 8 - 210, r.bottom() - 8), "UNCALIBRATED - estimated intrinsics",
                       fg=theme.WARN, align_center=False, size=8.5)
