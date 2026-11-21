"""'Limelight' page - live dashboard + push this app's calibration / field map to a Limelight.

After an upload the Limelight keeps the calibration on its own file system and uses it for
every AprilTag pipeline, so nothing has to run on the laptop during matches.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QThread, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QDoubleSpinBox,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..autolight import (
    AutoLightTuner,
    TuneResult,
    TuneSettings,
    apply_profile,
    delete_profile,
    load_profiles,
    revert,
    save_profile,
)
from ..camera import CameraConfig, VisionWorker, retire
from ..robotcode import CLASS_FILE, AdaptiveSettings, RobotCamera, generate, package_for, usage_snippet
from ..field_layout import FieldLayout, bundled_layouts
from ..geometry import euler_to_R, make_T
from ..limelight import (
    BACKUP_DIR,
    HOST_PRESETS,
    LimelightClient,
    LimelightError,
    LimelightPoller,
    LLResults,
    analyze_limelight_calibration,
    build_limelight_calibration,
    layout_to_fmap,
    pose6_to_T,
    save_backup,
    team_to_host,
)
from . import theme
from .field3d import Field3DView, RobotView
from .sweep_chart import SweepChart
from .widgets import Card, VideoView, compact_combos, draw_label, hint, value_label

MT1_COLOR = "#4cc3ff"
MT2_COLOR = "#c77dff"


class Job(QThread):
    """Run a blocking REST call off the UI thread."""

    done = Signal(object, object)  # result, error

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(), None)
        except Exception as exc:  # reported in the UI
            self.done.emit(None, exc)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.5g}"
    if isinstance(v, list):
        flat = [x for r in v for x in (r if isinstance(r, list) else [r])]
        return "[" + ", ".join(f"{x:.5g}" if isinstance(x, float) else str(x) for x in flat) + "]"
    return str(v)


class TuneJob(QThread):
    """Tune several Limelights one after another."""

    event = Signal(object)

    def __init__(self, hosts, settings, parent=None):
        super().__init__(parent)
        self.hosts, self.settings = hosts, settings
        self._stop = False
        self.results = []

    def stop(self):
        self._stop = True

    def run(self):
        for host in self.hosts:
            if self._stop:
                break
            self.event.emit({"kind": "start", "host": host})
            tuner = AutoLightTuner(host, self.settings,
                                   on_event=lambda kind, **kw: self.event.emit({"kind": kind, **kw}),
                                   should_stop=lambda: self._stop)
            self.results.append(tuner.run())
        self.event.emit({"kind": "all_done", "results": self.results})


class ChangesDialog(QDialog):
    def __init__(self, changes, warnings, host, res, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Upload calibration to Limelight")
        self.resize(780, 420)
        lay = QVBoxLayout(self)
        t = QLabel(f"<b>Upload to {host}</b> &nbsp; (Limelight resolution {res[0]}x{res[1]})")
        lay.addWidget(t)
        lay.addWidget(hint("These values in the Limelight's calibration JSON will change. The current Limelight "
                           f"calibration is backed up first to {BACKUP_DIR.name}/ and can be restored at any time."))
        for w in warnings:
            lab = QLabel(f"<span style='color:{theme.WARN}'>&#9888; {w}</span>")
            lab.setWordWrap(True)
            lay.addWidget(lab)
        tbl = QTableWidget(len(changes), 3)
        tbl.setHorizontalHeaderLabels(["Field", "On Limelight now", "New value"])
        tbl.verticalHeader().setVisible(False)
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        tbl.setWordWrap(True)
        for r, (path, old, new) in enumerate(changes):
            for c, v in enumerate((path, _fmt(old), _fmt(new))):
                it = QTableWidgetItem(v)
                if c == 2:
                    it.setForeground(QColor(theme.ACCENT))
                tbl.setItem(r, c, it)
        tbl.resizeRowsToContents()
        lay.addWidget(tbl, 1)
        bb = QDialogButtonBox()
        ok = bb.addButton("Back up && upload", QDialogButtonBox.ButtonRole.AcceptRole)
        ok.setObjectName("Primary")
        bb.addButton(QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


def _tab_scroll():
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(12, 12, 12, 12)
    lay.setSpacing(12)
    sa.setWidget(w)
    return sa, lay


class LimelightPage(QWidget):
    mode = "idle"
    status_changed = Signal(str, str)  # text, color

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.state = ctx.state
        self.client: LimelightClient | None = None
        self.poller: LimelightPoller | None = None
        self.stream: VisionWorker | None = None
        self.results: LLResults | None = None
        self.ll_cal = {}  # kind -> json
        self._jobs: list[Job] = []
        self._last_ui = 0.0

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        split = QSplitter(Qt.Orientation.Vertical)
        self.view = VideoView()
        self.view.placeholder = "Limelight stream - connect on the right"
        self.view.overlay = self._paint_overlay
        self.field = Field3DView()
        split.addWidget(self.view)
        split.addWidget(self.field)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)

        panel = QWidget()
        panel.setObjectName("Panel")
        panel.setMinimumWidth(410)
        panel.setMaximumWidth(520)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        pl.addWidget(self.tabs)
        root.addWidget(panel)
        self.tabs.addTab(self._build_connect_tab(), "Connect")
        self.tabs.addTab(self._build_targets_tab(), "Targets")
        self.tabs.addTab(self._build_calib_tab(), "Calibration")
        self.tabs.addTab(self._build_light_tab(), "Auto light")
        self.tabs.addTab(self._build_fmap_tab(), "Field map")
        self.tune_job: TuneJob | None = None
        self.tune_results: dict[str, TuneResult] = {}
        self._tune_points: dict[str, list] = {}

        self.state.layout_changed.connect(lambda: self.field.set_layout(self.state.layout))
        self.state.calibration_changed.connect(self._show_ours)
        if self.state.layout:
            self.field.set_layout(self.state.layout)
        self._show_ours()
        compact_combos(self)

    # ================================================================ tabs
    def _build_connect_tab(self):
        sa, lay = _tab_scroll()
        c = Card("Limelight", "Connect over the robot network, USB-C, or Wi-Fi radio. Uses the official "
                              "Limelight REST API (port 5807).")
        self.host = QComboBox()
        self.host.setEditable(True)
        for h, desc in HOST_PRESETS:
            self.host.addItem(h)
            self.host.setItemData(self.host.count() - 1, desc, Qt.ItemDataRole.ToolTipRole)
        c.add_row("Address", self.host)
        row = QHBoxLayout()
        self.team = QSpinBox()
        self.team.setRange(0, 25999)
        self.team.setSpecialValueText("-")
        use_team = QPushButton("Use 10.TE.AM.11")
        use_team.clicked.connect(self._use_team)
        row.addWidget(self.team, 1)
        row.addWidget(use_team)
        w = QWidget()
        w.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        c.add_row("Team number", w)
        btns = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setObjectName("Primary")
        self.connect_btn.clicked.connect(self.connect_ll)
        dis = QPushButton("Disconnect")
        dis.clicked.connect(self.disconnect_ll)
        btns.addWidget(self.connect_btn, 1)
        btns.addWidget(dis)
        c.add_layout(btns)
        self.conn_label = QLabel(f"<span style='color:{theme.MUTED}'>&#9679; Not connected</span>")
        self.conn_label.setWordWrap(True)
        c.add(self.conn_label)
        self.stream_check = QCheckBox("Show live video stream (port 5800)")
        self.stream_check.setChecked(True)
        self.stream_check.toggled.connect(self._toggle_stream)
        c.add(self.stream_check)
        web = QPushButton("Open Limelight web UI ↗")
        web.clicked.connect(lambda: self.client and QDesktopServices.openUrl(QUrl(self.client.ui_url)))
        c.add(web)
        lay.addWidget(c)

        st = Card("Device status")
        self.status_grid = QGridLayout()
        self.status_grid.setHorizontalSpacing(14)
        st.add_layout(self.status_grid)
        self.status_empty = hint("No data yet.")
        st.add(self.status_empty)
        lay.addWidget(st)

        use = Card("Calibrate a Limelight with this app",
                   "1. In the Limelight UI pick a pipeline with no overlays (Viewfinder) at the resolution you "
                   "compete with (LL3: 1280x960, LL3G/LL4: 1280x800).  2. Click below - the Limelight stream "
                   "becomes this app's camera.  3. Calibrate on the Calibration page.  4. Come back to the "
                   "Calibration tab here and upload.")
        b = QPushButton("Use Limelight stream as the app camera → Calibration")
        b.clicked.connect(self._use_as_camera)
        use.add(b)
        lay.addWidget(use)
        lay.addStretch(1)
        return sa

    def _build_targets_tab(self):
        sa, lay = _tab_scroll()
        mt = Card("Robot pose (Limelight, wpiblue)")
        self.mt = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        heads = ["", "x", "y", "z", "yaw"]
        for c, h in enumerate(heads):
            lab = QLabel(h)
            lab.setObjectName("Hint")
            grid.addWidget(lab, 0, c)
        for r, (key, name, col) in enumerate((("mt1", "MegaTag", MT1_COLOR), ("mt2", "MegaTag2", MT2_COLOR)), 1):
            lab = QLabel(f"<span style='color:{col}'>&#9679;</span> {name}")
            grid.addWidget(lab, r, 0)
            for c, k in enumerate(("x", "y", "z", "yaw"), 1):
                v = value_label()
                self.mt[(key, k)] = v
                grid.addWidget(v, r, c)
        mt.add_layout(grid)
        self.mt_meta = hint("")
        mt.add(self.mt_meta)
        self.show_mt2 = QCheckBox("Show MegaTag2 robot on the 3D field")
        self.show_mt2.setChecked(True)
        mt.add(self.show_mt2)
        lay.addWidget(mt)

        fc = Card("AprilTags seen by the Limelight")
        self.ll_count = QLabel("0")
        self.ll_count.setObjectName("Muted")
        fc.header_right.addWidget(self.ll_count)
        self.ll_table = QTableWidget(0, 5)
        self.ll_table.setHorizontalHeaderLabels(["ID", "tx°", "ty°", "ta %", "Dist m"])
        self.ll_table.verticalHeader().setVisible(False)
        self.ll_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.ll_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.ll_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.ll_table.setMinimumHeight(220)
        fc.add(self.ll_table)
        lay.addWidget(fc)

        lat = Card("Latency")
        self.lat_label = value_label()
        lat.add(self.lat_label)
        lay.addWidget(lat)
        lay.addStretch(1)
        return sa

    def _build_calib_tab(self):
        sa, lay = _tab_scroll()
        on = Card("Calibration on the Limelight")
        self.ll_cal_state = QLabel(f"<span style='color:{theme.MUTED}'>Not read yet</span>")
        self.ll_cal_state.setWordWrap(True)
        on.add(self.ll_cal_state)
        self.llv = self._intrinsics_grid(on)
        rd = QPushButton("Read from Limelight")
        rd.clicked.connect(self.read_calibration)
        on.add(rd)
        lay.addWidget(on)

        ours = Card("This app's calibration")
        self.our_state = QLabel()
        self.our_state.setWordWrap(True)
        ours.add(self.our_state)
        self.ourv = self._intrinsics_grid(ours)
        self.compare = hint("")
        ours.add(self.compare)
        lay.addWidget(ours)

        up = Card("Push to the Limelight",
                  "The calibration is written to the Limelight's file system (/cal-file). It survives reboots "
                  "and is used by every AprilTag pipeline on its own - no laptop needed at competition.")
        self.upload_btn = QPushButton("Upload calibration to Limelight")
        self.upload_btn.setObjectName("Primary")
        self.upload_btn.setMinimumHeight(36)
        self.upload_btn.clicked.connect(self.upload_calibration)
        up.add(self.upload_btn)
        r = QHBoxLayout()
        bk = QPushButton("Back up now")
        bk.clicked.connect(self.backup_now)
        rs = QPushButton("Restore backup…")
        rs.clicked.connect(self.restore_backup)
        r.addWidget(bk)
        r.addWidget(rs)
        up.add_layout(r)
        rm = QPushButton("Remove custom calibration (use factory default)")
        rm.setObjectName("Danger")
        rm.clicked.connect(self.remove_custom)
        up.add(rm)
        self.verify_label = QLabel("")
        self.verify_label.setWordWrap(True)
        up.add(self.verify_label)
        lay.addWidget(up)
        lay.addStretch(1)
        return sa

    def _build_light_tab(self):
        sa, lay = _tab_scroll()
        intro = Card("Automatic light tuning",
                     "Finds the shortest exposure at which every AprilTag in view is detected in (almost) every "
                     "frame with low noise - less motion blur while driving - plus a safety margin. Do it on the real "
                     "field (field calibration / practice time): robot still, tags in view at match distance. "
                     "The result is saved on each Limelight, so nothing runs on the laptop during matches.")
        lay.addWidget(intro)

        hosts = Card("Limelights")
        self.ll_list = QListWidget()
        self.ll_list.setFixedHeight(96)
        hosts.add(self.ll_list)
        row = QHBoxLayout()
        self.add_host = QLineEdit()
        self.add_host.setPlaceholderText("limelight-front.local / 10.TE.AM.12")
        self.add_host.returnPressed.connect(lambda: self._add_host(self.add_host.text()))
        b_add = QPushButton("Add")
        b_add.clicked.connect(lambda: self._add_host(self.add_host.text()))
        row.addWidget(self.add_host, 1)
        row.addWidget(b_add)
        hosts.add_layout(row)
        row2 = QHBoxLayout()
        b_conn = QPushButton("+ connected")
        b_conn.clicked.connect(lambda: self._add_host(self.client.host if self.client else self.host.currentText()))
        b_team = QPushButton("+ team .11 .12 .13")
        b_team.setToolTip("Adds 10.TE.AM.11, .12 and .13 (team number from the Connect tab)")
        b_team.clicked.connect(self._add_team_hosts)
        b_rm = QPushButton("Remove")
        b_rm.clicked.connect(lambda: [self.ll_list.takeItem(self.ll_list.row(i)) for i in self.ll_list.selectedItems()])
        row2.addWidget(b_conn)
        row2.addWidget(b_team)
        row2.addWidget(b_rm)
        hosts.add_layout(row2)
        lay.addWidget(hosts)

        st = Card("Tuning")
        self.t_min = self._dspin(0.01, 50, 0.1, 2, " ms")
        self.t_max = self._dspin(0.1, 100, 10.0, 1, " ms")
        rng = QHBoxLayout()
        rng.addWidget(self.t_min)
        dash = QLabel("to")
        dash.setObjectName("Muted")
        rng.addWidget(dash)
        rng.addWidget(self.t_max)
        w = QWidget()
        w.setLayout(rng)
        rng.setContentsMargins(0, 0, 0, 0)
        st.add_row("Exposure range", w)
        self.t_steps = QSpinBox()
        self.t_steps.setRange(5, 30)
        self.t_steps.setValue(12)
        st.add_row("Steps", self.t_steps)
        self.t_dwell = self._dspin(0.3, 5, 1.0, 1, " s / step")
        st.add_row("Measure", self.t_dwell)
        self.t_rel = self._dspin(50, 100, 95, 0, " % of frames")
        st.add_row("Reliability target", self.t_rel)
        self.t_margin = self._dspin(1.0, 3.0, 1.5, 2, " x")
        self.t_margin.setToolTip("Recommended exposure = shortest reliable exposure x margin")
        st.add_row("Safety margin", self.t_margin)
        self.t_save = QCheckBox("Save the result on each Limelight (persists after reboot)")
        self.t_save.setChecked(True)
        st.add(self.t_save)
        b = QHBoxLayout()
        self.tune_btn = QPushButton("Auto-tune lighting")
        self.tune_btn.setObjectName("Primary")
        self.tune_btn.setMinimumHeight(36)
        self.tune_btn.clicked.connect(self.start_tune)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(lambda: self.tune_job and self.tune_job.stop())
        b.addWidget(self.tune_btn, 1)
        b.addWidget(self.stop_btn)
        st.add_layout(b)
        self.tune_bar = QProgressBar()
        self.tune_bar.setValue(0)
        st.add(self.tune_bar)
        self.tune_status = hint("Idle. Settings are only written to disk at the end; Stop restores the original.")
        st.add(self.tune_status)
        lay.addWidget(st)

        ch = Card("Exposure sweep")
        self.chart_host = QComboBox()
        self.chart_host.currentTextChanged.connect(self._show_chart)
        ch.header_right.addWidget(self.chart_host)
        self.chart = SweepChart()
        ch.add(self.chart)
        self.show_table = QCheckBox("Show data table")
        ch.add(self.show_table)
        self.sweep_table = QTableWidget(0, 5)
        self.sweep_table.setHorizontalHeaderLabels(["Exposure ms", "Reliability", "Tags", "Jitter °", "Result"])
        self.sweep_table.verticalHeader().setVisible(False)
        self.sweep_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.sweep_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.sweep_table.setFixedHeight(180)
        self.sweep_table.setVisible(False)
        self.show_table.toggled.connect(self.sweep_table.setVisible)
        ch.add(self.sweep_table)
        lay.addWidget(ch)

        rs = Card("Results")
        self.res_table = QTableWidget(0, 4)
        self.res_table.setHorizontalHeaderLabels(["Limelight", "Exposure", "Tags / reliability", "Saved"])
        self.res_table.verticalHeader().setVisible(False)
        self.res_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.res_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.res_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.res_table.setFixedHeight(130)
        rs.add(self.res_table)
        self.res_msg = hint("")
        rs.add(self.res_msg)
        r = QHBoxLayout()
        rev = QPushButton("Revert selected")
        rev.setToolTip("Write back the exposure / gain / black level from before the tune")
        rev.clicked.connect(self._revert_selected)
        prof = QPushButton("Save as lighting profile…")
        prof.clicked.connect(self._save_profile)
        r.addWidget(rev)
        r.addWidget(prof)
        rs.add_layout(r)
        lay.addWidget(rs)

        pf = Card("Lighting profiles", "One profile per venue / field (e.g. 'District 1 - main field', "
                                       "'Home shop'). Apply re-sends the saved exposure to every Limelight in it.")
        self.profile_combo = QComboBox()
        pf.add(self.profile_combo)
        r = QHBoxLayout()
        ap = QPushButton("Apply profile")
        ap.setObjectName("Primary")
        ap.clicked.connect(self._apply_profile)
        dl = QPushButton("Delete")
        dl.setObjectName("Danger")
        dl.clicked.connect(self._delete_profile)
        r.addWidget(ap, 1)
        r.addWidget(dl)
        pf.add_layout(r)
        self.profile_msg = hint("")
        pf.add(self.profile_msg)
        lay.addWidget(pf)
        lay.addWidget(self._build_robot_code_card())
        lay.addStretch(1)
        self._reload_profiles()
        return sa

    def _build_robot_code_card(self):
        rc = Card("During the match - adaptive exposure (robot code)",
                  "The laptop is not on the robot during matches, so this part runs in your robot code "
                  "(Java / WPILib). From the robot pose it predicts which tags should be visible, compares with "
                  "what each Limelight detects, and nudges the exposure up (dark corners) or down (sun, "
                  "spotlights, blur while driving fast). Bounded, never saved to the Limelight's disk, and every "
                  "match starts from the tuned value.")
        self.rc_table = QTableWidget(0, 4)
        self.rc_table.setHorizontalHeaderLabels(["Limelight IP", "NT name", "Exposure key", "Baseline ms"])
        self.rc_table.verticalHeader().setVisible(False)
        self.rc_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.rc_table.setFixedHeight(120)
        self.rc_table.setToolTip("Double-click to edit. NT name = the Limelight's hostname "
                                 "(NetworkTables table, e.g. limelight-front). Use static IPs on the robot.")
        rc.add(self.rc_table)
        r = QHBoxLayout()
        fill = QPushButton("Fill from tune results")
        fill.clicked.connect(self._rc_fill)
        add = QPushButton("Add row")
        add.clicked.connect(lambda: self._rc_add_row(team_to_host(self.team.value()) if self.team.value() else
                                                     "10.0.0.11", "limelight", "exposure", 1.5))
        rm = QPushButton("Remove")
        rm.clicked.connect(lambda: [self.rc_table.removeRow(i) for i in
                                    sorted({x.row() for x in self.rc_table.selectedItems()}, reverse=True)])
        r.addWidget(fill, 1)
        r.addWidget(add)
        r.addWidget(rm)
        rc.add_layout(r)
        self.rc_min = self._dspin(0.2, 1.0, 0.6, 2, " x")
        self.rc_max = self._dspin(1.0, 6.0, 2.5, 2, " x")
        rng = QHBoxLayout()
        rng.addWidget(self.rc_min)
        to = QLabel("to")
        to.setObjectName("Muted")
        rng.addWidget(to)
        rng.addWidget(self.rc_max)
        w = QWidget()
        w.setLayout(rng)
        rng.setContentsMargins(0, 0, 0, 0)
        rc.add_row("Allowed range (x baseline)", w)
        self.rc_step = self._dspin(1.05, 2.0, 1.2, 2, " x per step")
        rc.add_row("Step", self.rc_step)
        self.rc_dist = self._dspin(1.0, 10.0, 4.5, 1, " m")
        rc.add_row("Expect tags up to", self.rc_dist)
        self.rc_hfov = self._dspin(20, 120, 63.3, 1, "° H")
        self.rc_vfov = self._dspin(15, 100, 49.7, 1, "° V")
        fov = QHBoxLayout()
        fov.addWidget(self.rc_hfov)
        fov.addWidget(self.rc_vfov)
        w2 = QWidget()
        w2.setLayout(fov)
        fov.setContentsMargins(0, 0, 0, 0)
        rc.add_row("Camera FOV", w2)
        self.rc_auto = QCheckBox("Adapt in autonomous")
        self.rc_auto.setChecked(True)
        self.rc_tele = QCheckBox("Adapt in teleop")
        self.rc_tele.setChecked(True)
        modes = QHBoxLayout()
        modes.addWidget(self.rc_auto)
        modes.addWidget(self.rc_tele)
        rc.add_layout(modes)
        b = QHBoxLayout()
        exp = QPushButton("Export Java file…")
        exp.setObjectName("Primary")
        exp.clicked.connect(self._rc_export)
        cp = QPushButton("Copy usage code")
        cp.clicked.connect(self._rc_copy_usage)
        b.addWidget(exp, 1)
        b.addWidget(cp)
        rc.add_layout(b)
        self.rc_msg = hint("Telemetry appears in NetworkTables under AdaptiveExposure/<name> "
                           "(AdvantageScope / Elastic); 'enabled' there is a kill switch.")
        rc.add(self.rc_msg)
        return rc

    # ---------------------------------------------------------- robot code
    def _rc_add_row(self, host, name, key, baseline_ms, integer=True):
        r = self.rc_table.rowCount()
        self.rc_table.insertRow(r)
        for c, v in enumerate((host, name, key, f"{baseline_ms:.2f}")):
            it = QTableWidgetItem(v)
            if c == 2:
                it.setData(Qt.ItemDataRole.UserRole, integer)
            self.rc_table.setItem(r, c, it)

    def _rc_fill(self):
        good = [r for r in self.tune_results.values() if r.recommended_ms and r.keys]
        if not good:
            self.rc_msg.setText("Run the auto-tune first (or use 'Add row' and type the values).")
            return
        info = self.ll_cal.get("_active") if self.ll_cal else None
        if info is not None:
            h, v, _ = info.as_calibration().fov_deg()
            self.rc_hfov.setValue(h)
            self.rc_vfov.setValue(v)

        def names():
            out = {}
            for r in good:
                name = "limelight"
                try:
                    st = LimelightClient(r.host, timeout=1.0).status() or {}
                    for k in ("name", "hostname", "cameraname", "device_name"):
                        if isinstance(st.get(k), str) and st[k]:
                            name = st[k]
                            break
                except LimelightError:
                    pass
                out[r.host] = name
            return out

        def done(res, err):
            self.rc_table.setRowCount(0)
            for r in good:
                integer = isinstance(r.keys.original.get(r.keys.exposure), int)
                self._rc_add_row(r.host, (res or {}).get(r.host, "limelight"), r.keys.exposure,
                                 r.recommended_ms, integer)
            self.rc_msg.setText(f"{len(good)} Limelight(s) filled. Check the NT names and that the IPs are the "
                                "robot's static IPs (not limelight.local).")

        self._run(names, done)

    def _rc_cameras(self):
        mount = self.ctx.pages["apriltag"].mount_values() if "apriltag" in self.ctx.pages else {}
        cams = []
        for r in range(self.rc_table.rowCount()):
            vals = [self.rc_table.item(r, c).text().strip() if self.rc_table.item(r, c) else "" for c in range(4)]
            try:
                ms = float(vals[3])
            except ValueError:
                raise ValueError(f"Row {r + 1}: baseline '{vals[3]}' is not a number") from None
            integer = self.rc_table.item(r, 2).data(Qt.ItemDataRole.UserRole) if self.rc_table.item(r, 2) else True
            cams.append(RobotCamera(name=vals[1] or "limelight", host=vals[0], exposure_key=vals[2] or "exposure",
                                    baseline_raw=ms / 0.01, integer_values=integer is not False,
                                    mount=dict(mount) if mount else RobotCamera("", "").mount,
                                    hfov=self.rc_hfov.value(), vfov=self.rc_vfov.value()))
        return cams

    def _rc_settings(self):
        return AdaptiveSettings(min_factor=self.rc_min.value(), max_factor=self.rc_max.value(),
                                step=self.rc_step.value(), max_tag_distance=self.rc_dist.value(),
                                adapt_auto=self.rc_auto.isChecked(), adapt_teleop=self.rc_tele.isChecked())

    def _rc_export(self):
        try:
            cams = self._rc_cameras()
            if not cams:
                raise ValueError("Add at least one Limelight (Fill from tune results / Add row).")
        except ValueError as exc:
            QMessageBox.information(self, "Robot code", str(exc))
            return
        names = [c.name for c in cams]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes and QMessageBox.question(
                self, "Robot code", f"Several rows use the NT name {', '.join(dupes)}. Each Limelight has its own "
                                    "NetworkTables name (its hostname, e.g. limelight-front). Export anyway?") \
                != QMessageBox.StandardButton.Yes:
            return
        folder = QFileDialog.getExistingDirectory(self, "Robot project folder (e.g. src/main/java/frc/robot/vision)",
                                                  str(Path.home()))
        if not folder:
            return
        pkg = package_for(Path(folder))
        path = Path(folder) / CLASS_FILE
        if path.exists() and QMessageBox.question(self, "Robot code", f"{path.name} exists - overwrite?") \
                != QMessageBox.StandardButton.Yes:
            return
        path.write_text(generate(cams, self._rc_settings(), pkg), encoding="utf-8")
        QGuiApplication.clipboard().setText(usage_snippet(cams))
        short = "/".join(path.parts[-4:])
        self.rc_msg.setText(f"<span style='color:{theme.ACCENT}'>&#10003; Wrote …/{short}</span><br>package {pkg} - "
                            "the usage code is on the clipboard. Check robotToCamera for each camera "
                            "(taken from the AprilTags page mount values).")
        self.rc_msg.setToolTip(str(path))
        self.ctx.status(f"Robot code exported: {path}")

    def _rc_copy_usage(self):
        try:
            cams = self._rc_cameras()
        except ValueError as exc:
            self.rc_msg.setText(str(exc))
            return
        if cams:
            QGuiApplication.clipboard().setText(usage_snippet(cams))
            self.rc_msg.setText("Usage code copied to the clipboard.")

    @staticmethod
    def _dspin(lo, hi, val, dec, suffix):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(dec)
        s.setValue(val)
        s.setSuffix(suffix)
        return s

    # ---------------------------------------------------------- auto light
    def hosts_to_tune(self) -> list[str]:
        out = []
        for i in range(self.ll_list.count()):
            it = self.ll_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                out.append(it.text())
        return out

    def _add_host(self, host: str):
        host = (host or "").strip()
        if not host:
            return
        if any(self.ll_list.item(i).text() == host for i in range(self.ll_list.count())):
            return
        it = QListWidgetItem(host)
        it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        it.setCheckState(Qt.CheckState.Checked)
        self.ll_list.addItem(it)
        self.add_host.clear()

    def _add_team_hosts(self):
        if self.team.value() <= 0:
            self.ctx.status("Enter the team number on the Connect tab first.")
            return
        base = team_to_host(self.team.value()).rsplit(".", 1)[0]
        for last in (11, 12, 13):
            self._add_host(f"{base}.{last}")

    def tune_settings(self) -> TuneSettings:
        return TuneSettings(min_ms=self.t_min.value(), max_ms=max(self.t_max.value(), self.t_min.value() * 2),
                            steps=self.t_steps.value(), dwell_s=self.t_dwell.value(),
                            min_reliability=self.t_rel.value() / 100.0, margin=self.t_margin.value(),
                            save=self.t_save.isChecked())

    def start_tune(self):
        if self.tune_job is not None:
            return
        if self.ll_list.count() == 0 and self.client is not None:
            self._add_host(self.client.host)
        hosts = self.hosts_to_tune()
        if not hosts:
            QMessageBox.information(self, "Auto light", "Add at least one Limelight (or connect on the Connect tab).")
            return
        s = self.tune_settings()
        self._tune_points = {h: [] for h in hosts}
        self.chart_host.blockSignals(True)
        self.chart_host.clear()
        self.chart_host.addItems(hosts)
        self.chart_host.blockSignals(False)
        self.chart.clear()
        self.chart.range_ms = (s.min_ms, s.max_ms)
        self._tune_total = len(hosts) * (s.steps + 3)
        self._tune_done_steps = 0
        self.tune_bar.setRange(0, self._tune_total)
        self.tune_bar.setValue(0)
        self.tune_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.tune_job = TuneJob(hosts, s, self)
        self.tune_job.event.connect(self._on_tune_event)
        self.tune_job.start()

    def _on_tune_event(self, ev: dict):
        kind = ev.get("kind")
        if kind == "status":
            self.tune_status.setText(ev["text"])
        elif kind == "start":
            self.chart_host.setCurrentText(ev["host"])
        elif kind == "keys":
            self.tune_status.setText(f"{ev['host']}: pipeline {ev['pipeline']} - {ev['keys'].describe()}")
        elif kind == "point":
            self._tune_points[ev["host"]] = ev["points"]
            self._tune_done_steps = min(self._tune_total, self._tune_done_steps + 1)
            self.tune_bar.setValue(self._tune_done_steps)
            if self.chart_host.currentText() == ev["host"]:
                self._show_chart(ev["host"])
        elif kind == "done":
            r: TuneResult = ev["result"]
            self.tune_results[r.host] = r
            self._tune_points[r.host] = r.points + ([r.verify] if r.verify else [])
            self._show_chart(self.chart_host.currentText())
            self._fill_results()
        elif kind == "all_done":
            job, self.tune_job = self.tune_job, None
            if job:
                job.wait(2000)
            self.tune_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.tune_bar.setValue(self.tune_bar.maximum())
            res = ev["results"]
            ok = sum(1 for r in res if r.ok)
            self.tune_status.setText(f"Done: {ok}/{len(res)} Limelight(s) tuned.")
            self.ctx.status(f"Auto light: {ok}/{len(res)} Limelight(s) tuned.")

    def _show_chart(self, host):
        pts = self._tune_points.get(host, [])
        r = self.tune_results.get(host)
        s = self.tune_settings()
        self.chart.set_data(pts, s.min_reliability, r.jitter_limit if r else None,
                            r.recommended_ms if r and r.recommended_ms else None, (s.min_ms, s.max_ms))
        rows = sorted(pts, key=lambda q: q.ms)
        self.sweep_table.setRowCount(len(rows))
        for i, q in enumerate(rows):
            vals = [f"{q.ms:.2f}", f"{q.reliability * 100:.0f}%", f"{q.median_tags:.0f}",
                    "-" if q.jitter_deg is None else f"{q.jitter_deg:.3f}",
                    ("✓ " if q.passed else "✗ ") + q.phase]
            for c, v in enumerate(vals):
                self.sweep_table.setItem(i, c, QTableWidgetItem(v))

    def _fill_results(self):
        self.res_table.setRowCount(len(self.tune_results))
        msgs = []
        for row, (host, r) in enumerate(self.tune_results.items()):
            v = r.verify
            vals = [host,
                    f"{r.recommended_ms:.2f} ms" if r.recommended_ms else "-",
                    f"{r.expected_tags} / {v.reliability * 100:.0f}%" if v else "-",
                    "✓ saved" if r.saved else ("live only" if r.recommended_ms else "✗")]
            for c, val in enumerate(vals):
                it = QTableWidgetItem(val)
                if c == 3:
                    it.setForeground(QColor(theme.ACCENT if r.saved else theme.WARN))
                self.res_table.setItem(row, c, it)
            msgs.append(f"<b>{host}</b>: {r.message}")
        self.res_msg.setText("<br>".join(msgs))

    def _selected_results(self):
        rows = sorted({i.row() for i in self.res_table.selectedItems()})
        hosts = list(self.tune_results)
        return [self.tune_results[hosts[r]] for r in rows if r < len(hosts)]

    def _revert_selected(self):
        sel = [r for r in self._selected_results() if r.keys is not None]
        if not sel:
            self.ctx.status("Select a tuned Limelight in the results table.")
            return

        def work():
            out = []
            for r in sel:
                try:
                    revert(r)
                    out.append(f"{r.host}: restored {r.keys.describe()}")
                except LimelightError as exc:
                    out.append(f"{r.host}: {exc}")
            return out

        self._run(work, lambda res, err: self.res_msg.setText(str(err) if err else "<br>".join(res)))

    def _save_profile(self):
        good = [r for r in self.tune_results.values() if r.recommended_raw is not None]
        if not good:
            self.ctx.status("Run the auto-tune first.")
            return
        name, ok = QInputDialog.getText(self, "Lighting profile", "Profile name (venue / field):",
                                        text=time.strftime("Field %Y-%m-%d"))
        if ok and name.strip():
            save_profile(name.strip(), good)
            self._reload_profiles(name.strip())
            self.profile_msg.setText(f"Saved '{name.strip()}' ({len(good)} Limelight(s)).")

    def _reload_profiles(self, select: str | None = None):
        self.profile_combo.clear()
        for name, entry in sorted(load_profiles().items()):
            n = len(entry.get("limelights", {}))
            self.profile_combo.addItem(f"{name}  ({n} LL, {entry.get('created', '')[:10]})", name)
        if select:
            i = self.profile_combo.findData(select)
            if i >= 0:
                self.profile_combo.setCurrentIndex(i)

    def _apply_profile(self):
        name = self.profile_combo.currentData()
        if not name:
            return
        self.profile_msg.setText(f"Applying '{name}'…")
        self._run(lambda: apply_profile(name),
                  lambda res, err: self.profile_msg.setText(str(err) if err else "<br>".join(res)))

    def _delete_profile(self):
        name = self.profile_combo.currentData()
        if name and QMessageBox.question(self, "Delete profile", f"Delete '{name}'?") == QMessageBox.StandardButton.Yes:
            delete_profile(name)
            self._reload_profiles()

    def _build_fmap_tab(self):
        sa, lay = _tab_scroll()
        c = Card("Field map (.fmap)",
                 "Limelight OS already ships with the official field maps. Upload only for a custom / practice "
                 "field, or to switch between the Welded and AndyMark layouts.")
        self.fmap_combo = QComboBox()
        for p in bundled_layouts():
            self.fmap_combo.addItem(p.stem, str(p))
        c.add(self.fmap_combo)
        r = QHBoxLayout()
        save = QPushButton("Save .fmap…")
        save.clicked.connect(self.save_fmap)
        upl = QPushButton("Upload to Limelight")
        upl.setObjectName("Primary")
        upl.clicked.connect(self.upload_fmap)
        r.addWidget(save)
        r.addWidget(upl, 1)
        c.add_layout(r)
        c.add(hint("Format: field-centre origin, same axes as botpose_wpiblue, 4x4 row-major transforms, "
                   "tag size in mm, family apriltag3_36h11_classic. After uploading, check that the Limelight's "
                   "MegaTag pose on the Targets tab matches where the robot really is."))
        lay.addWidget(c)
        lay.addStretch(1)
        return sa

    def _intrinsics_grid(self, card):
        vals = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        keys = [("res", "Resolution"), ("fx", "fx"), ("fy", "fy"), ("cx", "cx"), ("cy", "cy"),
                ("hfov", "HFOV"), ("aspect", "fy/fx"), ("rms", "RMS px"), ("dist", "Distortion")]
        for n, (k, label) in enumerate(keys[:-1]):
            r, col = divmod(n, 4)
            lab = QLabel(label)
            lab.setObjectName("Hint")
            v = value_label()
            vals[k] = v
            grid.addWidget(lab, r * 2, col)
            grid.addWidget(v, r * 2 + 1, col)
        card.add_layout(grid)
        d = value_label("")
        d.setWordWrap(True)
        vals["dist"] = d
        card.add_row("Distortion", d)
        return vals

    @staticmethod
    def _fill_grid(vals, calib):
        if calib is None:
            for v in vals.values():
                v.setText("-")
            return
        h, _v, _d = calib.fov_deg()
        vals["res"].setText(f"{calib.width}x{calib.height}")
        vals["fx"].setText(f"{calib.fx:.2f}")
        vals["fy"].setText(f"{calib.fy:.2f}")
        vals["cx"].setText(f"{calib.cx:.2f}")
        vals["cy"].setText(f"{calib.cy:.2f}")
        vals["hfov"].setText(f"{h:.2f}°")
        a = calib.pixel_aspect
        vals["aspect"].setText(f"<span style='color:{theme.ACCENT if 0.99 <= a <= 1.01 else theme.WARN}'>{a:.4f}</span>")
        vals["rms"].setText(f"{calib.rms:.3f}" if calib.rms == calib.rms else "-")
        vals["dist"].setText("  ".join(f"{x:.4f}" for x in np.asarray(calib.dist).ravel()))

    # ============================================================ connection
    def _use_team(self):
        if self.team.value() > 0:
            self.host.setCurrentText(team_to_host(self.team.value()))

    def connect_ll(self):
        self.disconnect_ll(quiet=True)
        host = self.host.currentText().strip()
        if not host:
            return
        self.client = LimelightClient(host)
        slow = "" if host.replace(".", "").isdigit() else \
            "<br><span style='color:#8c95a1'>Name lookup can take a few seconds - the IP is faster.</span>"
        self.conn_label.setText(f"<span style='color:{theme.WARN}'>&#9679; Connecting to {host}…</span>{slow}")
        self.status_changed.emit("LL connecting", theme.WARN)
        self.poller = LimelightPoller(host, parent=self)
        self.poller.results_ready.connect(self._on_results)
        self.poller.status_ready.connect(self._on_status)
        self.poller.connection.connect(self._on_connection)
        self.poller.start()

    def disconnect_ll(self, quiet=False):
        if self.poller is not None:
            p, self.poller = self.poller, None
            p.results_ready.disconnect()
            p.stop()
        self._stop_stream()
        self.field.set_robot("ll", None)
        self.field.set_robot("ll2", None)
        if not quiet:
            self.client = None
            self.conn_label.setText(f"<span style='color:{theme.MUTED}'>&#9679; Not connected</span>")
            self.status_changed.emit("LL offline", theme.MUTED)
            self.view.clear("Limelight disconnected")

    def _on_connection(self, ok: bool, msg: str):
        host = self.client.host if self.client else ""
        if ok:
            if self.client and self.poller:
                self.client._ip = self.poller.client._ip  # reuse the resolved address
            self.conn_label.setText(f"<span style='color:{theme.ACCENT}'>&#9679; Connected to {host}</span>")
            self.status_changed.emit(f"LL {host}", theme.ACCENT)
            self.ctx.status(f"Limelight connected: {host}")
            if self.stream_check.isChecked() and self.stream is None:
                self._start_stream()
            if not self.ll_cal:
                self.read_calibration(quiet=True)
        else:
            self.conn_label.setText(f"<span style='color:{theme.BAD}'>&#9679; {msg}</span>")
            self.status_changed.emit("LL offline", theme.BAD)

    def _toggle_stream(self, on):
        if on and self.client:
            self._start_stream()
        elif not on:
            self._stop_stream()
            self.view.clear("Stream hidden")

    def _start_stream(self):
        self._stop_stream()
        if not self.client:
            return
        self.view.clear(f"Opening stream {self.client.stream_url}…")
        w = VisionWorker(CameraConfig(name="Limelight stream", url=self.client.stream_url), self)
        w.packet_ready.connect(self._on_frame)
        w.failed.connect(lambda m: self.view.clear(f"{m}\nThe REST data still works without the stream."))
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
            self.view.set_image(pkt.image, self.results or False)

    def _use_as_camera(self):
        if not self.client:
            self.ctx.status("Connect to the Limelight first.")
            return
        self.stream_check.setChecked(False)  # one stream connection is enough
        self.ctx.use_stream_as_camera(self.client.stream_url, "Limelight " + self.client.host)

    # ================================================================ live
    def _on_status(self, st):
        while self.status_grid.count():
            it = self.status_grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        items = []
        if isinstance(st, dict):
            items = [(k, v) for k, v in st.items() if isinstance(v, (int, float, str)) and not isinstance(v, bool)]
        if self.results and self.results.hw:
            items += [(f"hw.{k}", v) for k, v in self.results.hw.items() if isinstance(v, (int, float, str))]
        self.status_empty.setVisible(not items)
        for n, (k, v) in enumerate(items[:16]):
            r, c = divmod(n, 2)
            lab = QLabel(str(k))
            lab.setObjectName("Hint")
            val = value_label(f"{v:.1f}" if isinstance(v, float) else str(v))
            self.status_grid.addWidget(lab, r, c * 2)
            self.status_grid.addWidget(val, r, c * 2 + 1)

    def _on_results(self, r: LLResults):
        self.results = r
        cam_T = None
        if r.botpose_blue:
            T = pose6_to_T(r.botpose_blue)
            cr = r.cam_robot if isinstance(r.cam_robot, list) and len(r.cam_robot) >= 6 else None
            if cr and any(abs(float(x)) > 1e-6 for x in cr[:6]):
                # visual only: mounting height / offset / yaw of the camera
                cam_T = T @ make_T(euler_to_R(0, 0, math.radians(float(cr[5]))), [float(cr[0]), float(cr[1]), float(cr[2])])
            else:
                cam_T = T @ make_T(np.eye(3), [0, 0, 0.5])
            ids = [f.id for f in r.fiducials]
            hfov = 63.3
            info = self.ll_cal.get("_active")
            if info is not None:
                hfov = info.as_calibration().fov_deg()[0]
            self.field.set_robot("ll", RobotView(T, cam_T, ids, MT1_COLOR, "LL MegaTag", hfov, hfov * 0.78))
        else:
            self.field.set_robot("ll", None)
        if r.botpose_mt2_blue and self.show_mt2.isChecked():
            self.field.set_robot("ll2", RobotView(pose6_to_T(r.botpose_mt2_blue), None, [], MT2_COLOR, "LL MegaTag2"))
        else:
            self.field.set_robot("ll2", None)
        self.field.set_visible([f.id for f in r.fiducials])

        now = time.monotonic()
        if now - self._last_ui < 0.1:
            return
        self._last_ui = now
        for key, pose in (("mt1", r.botpose_blue), ("mt2", r.botpose_mt2_blue)):
            vals = (pose[0], pose[1], pose[2], pose[5]) if pose else (None,) * 4
            for k, v in zip(("x", "y", "z", "yaw"), vals):
                self.mt[(key, k)].setText("-" if v is None else (f"{v:.1f}°" if k == "yaw" else f"{v:.3f}"))
        sd1 = r.stdev_mt1 if isinstance(r.stdev_mt1, list) and len(r.stdev_mt1) >= 6 else None
        sd2 = r.stdev_mt2 if isinstance(r.stdev_mt2, list) and len(r.stdev_mt2) >= 6 else None
        meta = f"tags {r.tagcount} | avg dist {r.avgdist:.2f} m | span {r.span:.2f} m | pipeline {r.pipeline} {r.ptype}"
        if sd1:
            meta += f"<br>std MT1: x {sd1[0]:.3f}  y {sd1[1]:.3f}  yaw {sd1[5]:.2f}°"
        if sd2:
            meta += f"<br>std MT2: x {sd2[0]:.3f}  y {sd2[1]:.3f}  yaw {sd2[5]:.2f}°"
        self.mt_meta.setText(meta)
        fids = sorted(r.fiducials, key=lambda f: f.id)
        self.ll_count.setText(f"{len(fids)} tags")
        self.ll_table.setRowCount(len(fids))
        for row, f in enumerate(fids):
            for c, v in enumerate((str(f.id), f"{f.tx:.2f}", f"{f.ty:.2f}", f"{f.ta:.2f}",
                                   "-" if f.distance is None else f"{f.distance:.2f}")):
                it = self.ll_table.item(row, c)
                if it is None:
                    it = QTableWidgetItem()
                    it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.ll_table.setItem(row, c, it)
                it.setText(v)
                if c == 0:
                    it.setForeground(QColor(theme.ACCENT))
        self.lat_label.setText(f"targeting {r.tl:.1f} ms + capture {r.cl:.1f} ms = {r.tl + r.cl:.1f} ms")

    def _paint_overlay(self, p, view):
        r = self.results
        rect = view.image_rect()
        if not r:
            return
        txt = f"LIMELIGHT  pipeline {r.pipeline}  |  {len(r.fiducials)} tags"
        if r.botpose_blue:
            b = r.botpose_blue
            txt += f"  |  x {b[0]:.2f}  y {b[1]:.2f}  yaw {b[5]:.1f}°"
        draw_label(p, QPointF(rect.x() + 8, rect.y() + 28), txt, fg=MT1_COLOR, align_center=False, size=10)

    # ========================================================== calibration
    def _run(self, fn, on_done):
        job = Job(fn, self)
        self._jobs.append(job)

        def finished(res, err):
            self._jobs.remove(job)
            on_done(res, err)

        job.done.connect(finished)
        job.start()

    def _need_client(self) -> bool:
        if self.client is None:
            QMessageBox.information(self, "Limelight", "Connect to a Limelight first (Connect tab).")
            return False
        return True

    def _fetch_all(self):
        out = {}
        for kind in ("file", "eeprom", "default"):
            try:
                out[kind] = self.client.calibration(kind)
            except LimelightError as exc:
                out[kind] = None
                out[f"{kind}_error"] = str(exc)
        try:
            out["hwreport"] = self.client.hwreport()
        except LimelightError:
            out["hwreport"] = None
        return out

    def read_calibration(self, quiet=False):
        if self.client is None:
            if not quiet:
                self._need_client()
            return
        self.ll_cal_state.setText(f"<span style='color:{theme.MUTED}'>Reading…</span>")

        def done(res, err):
            if err:
                self.ll_cal_state.setText(f"<span style='color:{theme.BAD}'>{err}</span>")
                return
            self._show_ll_cal(res)

        self._run(self._fetch_all, done)

    def _show_ll_cal(self, res: dict):
        self.ll_cal = dict(res)
        active, src = None, None
        for kind, label in (("file", "custom (file)"), ("eeprom", "custom (EEPROM)"), ("default", "factory default")):
            data = res.get(kind)
            if not data:
                continue
            try:
                info = analyze_limelight_calibration(data)
            except LimelightError:
                continue
            active, src = info, label
            break
        self.ll_cal["_active"] = active
        if active is None:
            keys = ", ".join(sorted(k for k in res if not k.endswith("_error") and res.get(k)))
            self.ll_cal_state.setText(f"<span style='color:{theme.WARN}'>Could not interpret the Limelight "
                                      f"calibration JSON (got: {keys or 'nothing'}).</span>")
            self._fill_grid(self.llv, None)
            return
        color = theme.ACCENT if src.startswith("custom") else theme.WARN
        extra = ""
        if res.get("file") and res.get("eeprom"):
            extra = (f"<br><span style='color:{theme.WARN}'>Both a file and an EEPROM calibration exist - "
                     "check in the Limelight UI which one is active.</span>")
        self.ll_cal_state.setText(f"<span style='color:{color}'>&#9679; Active: {src}</span>"
                                  f"{' (resolution inferred)' if active.res_inferred else ''}{extra}")
        self._fill_grid(self.llv, active.as_calibration())
        self._show_ours()

    def _show_ours(self):
        c = self.state.calibration
        if c is None:
            self.our_state.setText(f"<span style='color:{theme.WARN}'>&#9679; No calibration yet - "
                                   "use the Calibration page.</span>")
            self._fill_grid(self.ourv, None)
            self.compare.setText("")
            return
        self.our_state.setText(f"<span style='color:{theme.ACCENT}'>&#9679; {c.camera_name or 'camera'}</span> "
                               f"&nbsp; {c.num_images} images, {c.created}")
        self._fill_grid(self.ourv, c)
        info = self.ll_cal.get("_active") if self.ll_cal else None
        if info is not None:
            llc = info.as_calibration()
            if abs(llc.width / llc.height - c.width / c.height) < 0.01:
                ours = c.for_resolution(llc.width, llc.height)
                dfx = 100 * (ours.fx - llc.fx) / llc.fx
                dh = ours.fov_deg()[0] - llc.fov_deg()[0]
                self.compare.setText(f"vs Limelight at {llc.width}x{llc.height}: fx {dfx:+.2f}% | HFOV {dh:+.2f}° "
                                     f"| cx {ours.cx - llc.cx:+.1f} px | cy {ours.cy - llc.cy:+.1f} px")
            else:
                self.compare.setText(f"<span style='color:{theme.WARN}'>Different aspect ratio than the Limelight "
                                     f"({llc.width}x{llc.height}) - calibrate from the Limelight stream.</span>")

    def backup_now(self):
        if not self._need_client():
            return
        host = self.client.host

        def work():
            data = self._fetch_all()
            paths = [str(save_backup(host, k, data[k])) for k in ("file", "eeprom", "default") if data.get(k)]
            return data, paths

        def done(res, err):
            if err:
                QMessageBox.warning(self, "Backup", str(err))
                return
            data, paths = res
            self._show_ll_cal(data)
            self.ctx.status(f"Backed up {len(paths)} calibration file(s) to {BACKUP_DIR}")

        self._run(work, done)

    def upload_calibration(self):
        if not self._need_client():
            return
        calib = self.state.calibration
        if calib is None:
            QMessageBox.information(self, "Upload", "Calibrate on the Calibration page first.")
            return
        if calib.rms == calib.rms and calib.rms > 1.0:
            if QMessageBox.question(self, "Upload", f"This calibration has a high error ({calib.rms:.2f} px). "
                                                    "Upload anyway?") != QMessageBox.StandardButton.Yes:
                return
        self.upload_btn.setEnabled(False)
        self.verify_label.setText("Reading the Limelight's current calibration…")

        def got(res, err):
            self.upload_btn.setEnabled(True)
            if err:
                self.verify_label.setText(f"<span style='color:{theme.BAD}'>{err}</span>")
                return
            self._show_ll_cal(res)
            template = res.get("file") if res.get("file") else res.get("default")
            if not template:
                self.verify_label.setText(f"<span style='color:{theme.BAD}'>The Limelight returned no calibration "
                                          "to use as a template (cal-file and cal-default are empty).</span>")
                return
            try:
                new, changes, warnings, llres = build_limelight_calibration(template, calib)
            except LimelightError as exc:
                path = save_backup(self.client.host, "template", template)
                self.verify_label.setText(f"<span style='color:{theme.BAD}'>{exc}</span><br>"
                                          f"Template saved to {path.name} for inspection.")
                return
            dlg = ChangesDialog(changes, warnings, self.client.host, llres, self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.verify_label.setText("Upload cancelled.")
                return
            self._do_upload(res, new, calib, llres)

        self._run(self._fetch_all, got)

    def _do_upload(self, current: dict, new: dict, calib, llres):
        host = self.client.host
        self.upload_btn.setEnabled(False)
        self.verify_label.setText("Backing up and uploading…")

        def work():
            backups = [save_backup(host, k, current[k]) for k in ("file", "eeprom", "default") if current.get(k)]
            self.client.upload_cal_file(new)
            time.sleep(0.5)
            readback = self.client.calibration("file")
            return backups, readback

        def done(res, err):
            self.upload_btn.setEnabled(True)
            if err:
                self.verify_label.setText(f"<span style='color:{theme.BAD}'>Upload failed: {err}</span>")
                return
            backups, readback = res
            expected = calib.for_resolution(*llres)
            try:
                info = analyze_limelight_calibration(readback)
                ok = np.allclose(info.K, expected.K, rtol=1e-4, atol=1e-3)
            except LimelightError:
                ok = False
            if ok:
                self.verify_label.setText(
                    f"<span style='color:{theme.ACCENT}'>&#10003; Uploaded and verified.</span> The Limelight now "
                    f"uses fx {expected.fx:.1f}, fy {expected.fy:.1f} at {llres[0]}x{llres[1]}. "
                    f"Backups: {', '.join(p.name for p in backups) or 'none'}")
                self.ctx.status("Calibration uploaded to the Limelight and verified.")
            else:
                self.verify_label.setText(
                    f"<span style='color:{theme.WARN}'>Uploaded, but reading /cal-file back did not return the new "
                    "values. Check the Calibration tab of the Limelight web UI (green header = custom calibration "
                    "active).</span>")
            self.read_calibration(quiet=True)

        self._run(work, done)

    def restore_backup(self):
        if not self._need_client():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Restore Limelight calibration", str(BACKUP_DIR), "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Restore", str(exc))
            return
        if QMessageBox.question(self, "Restore", f"Upload {Path(path).name} as the Limelight's calibration "
                                                 f"(/cal-file) on {self.client.host}?") != QMessageBox.StandardButton.Yes:
            return
        host = self.client.host

        def work():
            cur = self.client.calibration("file")
            if cur:
                save_backup(host, "file-before-restore", cur)
            self.client.upload_cal_file(data)
            return True

        self._run(work, lambda _r, e: (self.verify_label.setText(
            f"<span style='color:{theme.BAD}'>{e}</span>" if e else
            f"<span style='color:{theme.ACCENT}'>&#10003; Restored {Path(path).name}</span>"),
            self.read_calibration(quiet=True)))

    def remove_custom(self):
        if not self._need_client():
            return
        if QMessageBox.question(self, "Remove custom calibration",
                                f"Delete the custom file calibration on {self.client.host}? It is backed up first, "
                                "and the Limelight goes back to its default calibration.") != QMessageBox.StandardButton.Yes:
            return
        host = self.client.host

        def work():
            cur = self.client.calibration("file")
            if cur:
                save_backup(host, "file-before-delete", cur)
            self.client.delete_cal_file()
            return True

        self._run(work, lambda _r, e: (self.verify_label.setText(
            f"<span style='color:{theme.BAD}'>{e}</span>" if e else "Custom calibration removed (backup saved)."),
            self.read_calibration(quiet=True)))

    # ============================================================ field map
    def _fmap_layout(self) -> FieldLayout | None:
        try:
            return FieldLayout.load(self.fmap_combo.currentData())
        except Exception as exc:
            QMessageBox.warning(self, "Field map", str(exc))
            return None

    def save_fmap(self):
        layout = self._fmap_layout()
        if layout is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Limelight field map",
                                              str(Path.home() / f"{Path(self.fmap_combo.currentData()).stem}.fmap"),
                                              "Limelight field map (*.fmap)")
        if path:
            Path(path).write_text(json.dumps(layout_to_fmap(layout), indent=1), encoding="utf-8")
            self.ctx.status(f"Field map saved: {path}")

    def upload_fmap(self):
        if not self._need_client():
            return
        layout = self._fmap_layout()
        if layout is None:
            return
        if QMessageBox.question(self, "Upload field map",
                                f"Upload '{layout.name}' ({len(layout.tags)} tags) to {self.client.host}?\n\n"
                                "Limelight already includes the official map - only do this for a custom field "
                                "or to switch Welded/AndyMark.") != QMessageBox.StandardButton.Yes:
            return
        fmap = layout_to_fmap(layout)
        self._run(lambda: self.client.upload_fieldmap(fmap), lambda _r, e: self.ctx.status(
            f"Field map upload failed: {e}" if e else f"Field map '{layout.name}' uploaded to the Limelight."))

    # ================================================================ misc
    def on_packet(self, _pkt):
        pass  # the main camera is not shown here

    def shutdown(self):
        if self.tune_job is not None:
            self.tune_job.stop()  # tuner restores the original exposure on stop
            self.tune_job.wait(8000)
        self.disconnect_ll(quiet=True)
        for j in list(self._jobs):
            j.wait(3000)
