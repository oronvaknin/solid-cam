"""'Calibration' page - a Limelight-style ChArUco camera calibration, inside the app."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QMarginsF, QPointF, QRectF, QSize, QSizeF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPageLayout, QPageSize, QPainter, QPdfWriter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..calibration import (
    ARUCO_DICTS,
    BOARD_PRESETS,
    BoardConfig,
    CalibrationSession,
    CameraCalibration,
    CharucoDetection,
    CharucoPipeline,
    default_calibration_path,
)
from ..camera import bgr_to_qimage
from . import theme
from .capture3d import CapturePosesView
from .widgets import Card, VideoView, compact_combos, draw_label, hint, scroll_panel, value_label

RECOMMENDED_CAPTURES = 25


def rms_verdict(rms: float):
    if rms < 0.35:
        return "Excellent", theme.ACCENT
    if rms < 0.7:
        return "Good", theme.ACCENT
    if rms < 1.0:
        return "OK", theme.WARN
    return "Poor - recapture", theme.BAD


class CalibrateJob(QThread):
    done = Signal(object)

    def __init__(self, session, board, opts):
        super().__init__()
        self.session, self.board, self.opts = session, board, opts

    def run(self):
        try:
            self.done.emit(self.session.calibrate(self.board, **self.opts))
        except Exception as exc:  # report to UI
            self.done.emit(exc)


class CalibrationPage(QWidget):
    mode = "charuco"

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.state = ctx.state
        self.board_cfg = BoardConfig()
        self.pipeline = CharucoPipeline(self.board_cfg)
        self.session = CalibrationSession()
        self.show_undistorted = False
        self._undist_cache = (None, None)
        self.last_det: CharucoDetection | None = None
        self.last_frame = None
        self._prev_det: CharucoDetection | None = None
        self._still_since = None
        self._last_capture_t = 0.0
        self._last_ui = 0.0
        self._flash_until = 0.0
        self._job: CalibrateJob | None = None
        self._coverage_pts = np.zeros((0, 2), np.float32)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(0)
        self.view = VideoView()
        self.view.overlay = self._paint_overlay
        self.view.show_crosshair = False
        self.poses_view = CapturePosesView()
        self.left_stack = QStackedWidget()
        self.left_stack.addWidget(self.view)
        self.left_stack.addWidget(self.poses_view)
        seg = QWidget()
        seg.setObjectName("SegBar")
        sl = QHBoxLayout(seg)
        sl.setContentsMargins(10, 6, 10, 6)
        sl.setSpacing(4)
        self.seg_live = QPushButton("Live camera")
        self.seg_3d = QPushButton("3D board poses")
        for i, b in enumerate((self.seg_live, self.seg_3d)):
            b.setObjectName("Seg")
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.clicked.connect(lambda _=False, i=i: self._show_left(i))
            sl.addWidget(b)
        self.seg_live.setChecked(True)
        sl.addStretch(1)
        left.addWidget(seg)
        left.addWidget(self.left_stack, 1)

        gal = QWidget()
        gal.setObjectName("Panel")
        gl = QVBoxLayout(gal)
        gl.setContentsMargins(10, 6, 10, 8)
        head = QHBoxLayout()
        self.gallery_title = QLabel("Captures (0)")
        self.gallery_title.setObjectName("CardTitle")
        head.addWidget(self.gallery_title)
        head.addStretch(1)
        self.del_btn = QPushButton("Delete selected")
        self.del_btn.clicked.connect(self._delete_selected)
        clear = QPushButton("Clear all")
        clear.setObjectName("Danger")
        clear.clicked.connect(self._clear_all)
        head.addWidget(self.del_btn)
        head.addWidget(clear)
        gl.addLayout(head)
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListView.ViewMode.IconMode)
        self.gallery.setFlow(QListView.Flow.LeftToRight)
        self.gallery.setWrapping(False)
        self.gallery.setIconSize(QSize(144, 81))
        self.gallery.setFixedHeight(128)
        self.gallery.setMovement(QListView.Movement.Static)
        self.gallery.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.gallery.setSpacing(4)
        gl.addWidget(self.gallery)
        left.addWidget(gal)
        lw = QWidget()
        lw.setLayout(left)
        root.addWidget(lw, 1)

        panel, lay = scroll_panel(400)
        root.addWidget(panel)
        lay.addWidget(self._build_board_card())
        lay.addWidget(self._build_capture_card())
        lay.addWidget(self._build_calibrate_card())
        lay.addWidget(self._build_result_card())
        lay.addWidget(self._build_limelight_card())
        lay.addStretch(1)

        self.state.calibration_changed.connect(self._show_result)
        self._show_result()
        compact_combos(self)

    # ============================================================ build UI
    def _build_board_card(self):
        c = Card("1  Calibration board",
                 "Print a ChArUco board, glue it to something flat and rigid, then measure the real "
                 "square size with a ruler / caliper.")
        self.preset = QComboBox()
        self.preset.addItems(list(BOARD_PRESETS) + ["Custom"])
        self.preset.currentTextChanged.connect(self._preset_changed)
        c.add(self.preset)
        self.dict_combo = QComboBox()
        self.dict_combo.addItems(list(ARUCO_DICTS))
        c.add_row("Dictionary", self.dict_combo)
        self.sx = QSpinBox()
        self.sx.setRange(3, 30)
        self.sy = QSpinBox()
        self.sy.setRange(3, 30)
        row = QHBoxLayout()
        row.addWidget(self.sx)
        x = QLabel("x")
        x.setObjectName("Muted")
        row.addWidget(x)
        row.addWidget(self.sy)
        row.addStretch(1)
        w = QWidget()
        w.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        c.add_row("Squares (X x Y)", w)
        self.sq = QDoubleSpinBox()
        self.sq.setRange(1, 500)
        self.sq.setDecimals(2)
        self.sq.setSuffix(" mm")
        self.mk = QDoubleSpinBox()
        self.mk.setRange(1, 500)
        self.mk.setDecimals(2)
        self.mk.setSuffix(" mm")
        c.add_row("Square size", self.sq)
        c.add_row("Marker size", self.mk)
        self.legacy = QCheckBox("Legacy pattern (old calib.io / OpenCV < 4.6 boards)")
        c.add(self.legacy)
        self.board_desc = hint("")
        c.add(self.board_desc)
        btns = QHBoxLayout()
        pdf = QPushButton("Save printable PDF…")
        pdf.setObjectName("Primary")
        pdf.clicked.connect(self._save_pdf)
        png = QPushButton("PNG…")
        png.clicked.connect(self._save_png)
        btns.addWidget(pdf, 1)
        btns.addWidget(png)
        c.add_layout(btns)
        c.add(hint("Print at 100% / 'Actual size' (never 'Fit to page'). The PDF includes a 100 mm "
                   "ruler to check the print scale."))
        for wdg in (self.dict_combo,):
            wdg.currentTextChanged.connect(self._board_changed)
        for wdg in (self.sx, self.sy, self.sq, self.mk):
            wdg.valueChanged.connect(self._board_changed)
        self.legacy.toggled.connect(self._board_changed)
        self._set_board_widgets(self.board_cfg)
        return c

    def _build_capture_card(self):
        c = Card("2  Capture", "Move the board around the WHOLE image: centre, edges, corners, near, far and "
                              "tilted ~30-45°. Hold still for each capture.")
        self.det_bar = QProgressBar()
        self.det_bar.setFormat("Board: %v / %m corners")
        self.det_bar.setRange(0, self.board_cfg.total_corners)
        self.det_bar.setValue(0)
        c.add(self.det_bar)
        self.legacy_btn = QPushButton("⚠ Markers found but no corners - click to toggle 'Legacy pattern'")
        self.legacy_btn.setStyleSheet(f"color: {theme.WARN}; text-align: left;")
        self.legacy_btn.setVisible(False)
        self.legacy_btn.clicked.connect(lambda: self.legacy.setChecked(not self.legacy.isChecked()))
        c.add(self.legacy_btn)
        stats = QGridLayout()
        self.v_sharp = value_label()
        self.v_motion = value_label()
        for i, (lab, v) in enumerate((("Sharpness", self.v_sharp), ("Motion", self.v_motion))):
            l = QLabel(lab)
            l.setObjectName("Hint")
            stats.addWidget(l, 0, i)
            stats.addWidget(v, 1, i)
        c.add_layout(stats)
        self.capture_btn = QPushButton("Capture   [Space]")
        self.capture_btn.setObjectName("Primary")
        self.capture_btn.setMinimumHeight(36)
        self.capture_btn.clicked.connect(self.capture)
        c.add(self.capture_btn)
        self.auto = QCheckBox("Auto-capture when the board is still and in a new position")
        self.auto.setChecked(True)
        c.add(self.auto)
        self.min_corners = QSpinBox()
        self.min_corners.setRange(6, 500)
        self.min_corners.setValue(12)
        c.add_row("Min corners / capture", self.min_corners)
        self.interval = QDoubleSpinBox()
        self.interval.setRange(0.3, 10)
        self.interval.setValue(1.0)
        self.interval.setSuffix(" s")
        c.add_row("Auto-capture interval", self.interval)
        self.count_bar = QProgressBar()
        self.count_bar.setRange(0, RECOMMENDED_CAPTURES)
        self.count_bar.setFormat(f"%v / {RECOMMENDED_CAPTURES} captures")
        c.add_row("Captures", self.count_bar)
        self.cov_bar = QProgressBar()
        self.cov_bar.setRange(0, 100)
        self.cov_bar.setFormat("%v% of image covered")
        c.add_row("Coverage", self.cov_bar)
        self.show_cov = QCheckBox("Show captured corners on video")
        self.show_cov.setChecked(True)
        c.add(self.show_cov)
        return c

    def _build_calibrate_card(self):
        c = Card("3  Calibrate")
        self.model = QComboBox()
        self.model.addItem("Standard (k1 k2 p1 p2 k3)", False)
        self.model.addItem("Rational - wide-angle lenses (8 coeffs)", True)
        c.add_row("Distortion model", self.model)
        self.fix_aspect = QCheckBox("Square pixels (fix fx = fy)")
        c.add(self.fix_aspect)
        self.calib_btn = QPushButton("Calibrate")
        self.calib_btn.setObjectName("Primary")
        self.calib_btn.setMinimumHeight(36)
        self.calib_btn.clicked.connect(self.run_calibration)
        c.add(self.calib_btn)
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setVisible(False)
        c.add(self.busy)
        return c

    def _build_result_card(self):
        c = Card("Result")
        self.rms_label = QLabel("-")
        self.rms_label.setObjectName("Big")
        self.rms_verdict = QLabel("No calibration yet")
        self.rms_verdict.setObjectName("Muted")
        row = QHBoxLayout()
        row.addWidget(self.rms_label)
        col = QVBoxLayout()
        cap = QLabel("RMS reprojection error")
        cap.setObjectName("Hint")
        col.addWidget(cap)
        col.addWidget(self.rms_verdict)
        row.addLayout(col, 1)
        c.add_layout(row)
        self.res = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        keys = [("fx", "fx"), ("fy", "fy"), ("res", "Resolution"), ("cx", "cx"), ("cy", "cy"),
                ("imgs", "Images"), ("hfov", "HFOV"), ("vfov", "VFOV"), ("dfov", "DFOV"),
                ("aspect", "Pixel aspect fy/fx")]
        for n, (k, label) in enumerate(keys):
            r, col_ = divmod(n, 3)
            l = QLabel(label)
            l.setObjectName("Hint")
            v = value_label()
            self.res[k] = v
            grid.addWidget(l, r * 2, col_)
            grid.addWidget(v, r * 2 + 1, col_)
        c.add_layout(grid)
        self.dist_label = value_label("")
        self.dist_label.setWordWrap(True)
        c.add_row("Distortion", self.dist_label)
        self.path_label = hint("")
        c.add(self.path_label)
        self.undist = QCheckBox("Show undistorted preview (straight lines should look straight)")
        self.undist.toggled.connect(self._toggle_undistort)
        c.add(self.undist)
        r1 = QHBoxLayout()
        self.prune_btn = QPushButton("Drop worst 15% + recalibrate")
        self.prune_btn.clicked.connect(self._prune_and_recalibrate)
        r1.addWidget(self.prune_btn)
        c.add_layout(r1)
        r2 = QHBoxLayout()
        save = QPushButton("Save as…")
        save.clicked.connect(self._save_as)
        load = QPushButton("Load…")
        load.clicked.connect(self._load)
        r2.addWidget(save)
        r2.addWidget(load)
        c.add_layout(r2)
        c.add(hint("A good calibration is usually below 0.5 px RMS with 20-40 well spread captures. "
                   "The result is used immediately by the AprilTags page and saved automatically."))
        return c

    def _build_limelight_card(self):
        c = Card("Calibrating a Limelight?",
                 "Use the Limelight's own stream as the camera (Limelight page → 'Use Limelight stream'), "
                 "calibrate here, then upload from the Limelight page. The Limelight stores the calibration and "
                 "uses it in every match - the app does not need to be running.")
        b = QPushButton("Open the Limelight page →")
        b.clicked.connect(lambda: self.ctx.goto_page("limelight"))
        c.add(b)
        return c

    def _show_left(self, i: int):
        if i == 1:
            self._refresh_poses()
        self.left_stack.setCurrentIndex(i)

    def _refresh_poses(self):
        size = self.session.image_size or self.state.frame_size
        hfov, vfov = 70.0, 45.0
        if size and size[0]:
            hfov, vfov, _ = self.state.intrinsics(*size).fov_deg()
        self.poses_view.set_data(self.session.captures, self.pipeline.board, hfov, vfov)

    # ========================================================= board config
    def _set_board_widgets(self, cfg: BoardConfig):
        for w in (self.dict_combo, self.sx, self.sy, self.sq, self.mk, self.legacy):
            w.blockSignals(True)
        self.dict_combo.setCurrentText(cfg.dictionary)
        self.sx.setValue(cfg.squares_x)
        self.sy.setValue(cfg.squares_y)
        self.sq.setValue(cfg.square_mm)
        self.mk.setValue(cfg.marker_mm)
        self.legacy.setChecked(cfg.legacy)
        for w in (self.dict_combo, self.sx, self.sy, self.sq, self.mk, self.legacy):
            w.blockSignals(False)
        self._apply_board(cfg)

    def set_board_config(self, cfg: BoardConfig):
        for name, preset in BOARD_PRESETS.items():
            if preset == cfg:
                self.preset.blockSignals(True)
                self.preset.setCurrentText(name)
                self.preset.blockSignals(False)
                break
        else:
            self.preset.blockSignals(True)
            self.preset.setCurrentText("Custom")
            self.preset.blockSignals(False)
        self._set_board_widgets(cfg)

    def _preset_changed(self, name):
        if name in BOARD_PRESETS:
            p = BOARD_PRESETS[name]
            self._set_board_widgets(BoardConfig(**p.__dict__))

    def _board_changed(self, *_):
        cfg = BoardConfig(self.sx.value(), self.sy.value(), self.sq.value(), self.mk.value(),
                          self.dict_combo.currentText(), self.legacy.isChecked())
        if cfg.marker_mm >= cfg.square_mm:
            self.board_desc.setText(f"<span style='color:{theme.BAD}'>Marker must be smaller than the square.</span>")
            return
        match = next((n for n, p in BOARD_PRESETS.items() if p == cfg), "Custom")
        self.preset.blockSignals(True)
        self.preset.setCurrentText(match)
        self.preset.blockSignals(False)
        self._apply_board(cfg)

    def _apply_board(self, cfg: BoardConfig):
        old = self.board_cfg
        structural = (old.squares_x, old.squares_y, old.dictionary, old.legacy) != \
                     (cfg.squares_x, cfg.squares_y, cfg.dictionary, cfg.legacy)
        self.board_cfg = cfg
        try:
            self.pipeline = CharucoPipeline(cfg)
        except cv2.error as exc:
            self.board_desc.setText(f"<span style='color:{theme.BAD}'>Invalid board: {exc}</span>")
            return
        self.board_desc.setText(cfg.describe())
        if structural and self.session.captures:
            self.session.clear()
            self._rebuild_gallery()
            self.ctx.status("Board layout changed - previous captures cleared.")

    # ============================================================ printing
    def _save_pdf(self):
        cfg = self.board_cfg
        default = str(Path.home() / f"charuco_{cfg.squares_x}x{cfg.squares_y}_{cfg.square_mm:g}mm.pdf")
        path, _ = QFileDialog.getSaveFileName(self, "Save printable board", default, "PDF (*.pdf)")
        if not path:
            return
        bw, bh = cfg.size_mm
        margin, label_h = 12.0, 26.0
        need_w, need_h = bw + 2 * margin, bh + 2 * margin + label_h
        if need_w <= 297 and need_h <= 210:
            page = QPageSize(QPageSize.PageSizeId.A4)
            orient = QPageLayout.Orientation.Landscape
            pw, ph = 297.0, 210.0
        elif need_w <= 210 and need_h <= 297:
            page = QPageSize(QPageSize.PageSizeId.A4)
            orient = QPageLayout.Orientation.Portrait
            pw, ph = 210.0, 297.0
        else:
            page = QPageSize(QSizeF(need_w, need_h), QPageSize.Unit.Millimeter, "ChArUco")
            orient = QPageLayout.Orientation.Portrait
            pw, ph = need_w, need_h
        writer = QPdfWriter(path)
        writer.setResolution(300)
        writer.setPageLayout(QPageLayout(page, orient, QMarginsF(0, 0, 0, 0), QPageLayout.Unit.Millimeter))
        writer.setTitle("ChArUco calibration board")
        u = 300 / 25.4  # device units per mm
        px_per_square = 120
        img = cfg.make_board().generateImage((cfg.squares_x * px_per_square, cfg.squares_y * px_per_square),
                                             marginSize=0, borderBits=1)
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format.Format_Grayscale8).copy()
        p = QPainter(writer)
        x0 = (pw - bw) / 2
        y0 = margin
        p.drawImage(QRectF(x0 * u, y0 * u, bw * u, bh * u), qimg)
        p.setPen(QPen(QColor("black"), 0.3 * u))
        ry = (y0 + bh + 9) * u
        p.drawLine(QPointF(x0 * u, ry), QPointF((x0 + 100) * u, ry))
        for k in range(0, 101, 10):
            tick = 3 if k % 50 == 0 else 1.5
            p.drawLine(QPointF((x0 + k) * u, ry), QPointF((x0 + k) * u, ry - tick * u))
        f = QFont("Arial")
        f.setPointSizeF(9)
        p.setFont(f)
        p.drawText(QPointF((x0 + 103) * u, ry + 1 * u), "<- this line must measure exactly 100 mm")
        p.drawText(QPointF(x0 * u, (y0 + bh + 17) * u), cfg.describe())
        p.end()
        self.ctx.status(f"Board saved: {path}")

    def _save_png(self):
        cfg = self.board_cfg
        default = str(Path.home() / f"charuco_{cfg.squares_x}x{cfg.squares_y}_{cfg.square_mm:g}mm_300dpi.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save board image", default, "PNG (*.png)")
        if not path:
            return
        dpi = 300
        img = cfg.render(px_per_mm=dpi / 25.4, margin_mm=10)
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format.Format_Grayscale8).copy()
        qimg.setDotsPerMeterX(int(dpi / 0.0254))
        qimg.setDotsPerMeterY(int(dpi / 0.0254))
        qimg.save(path)
        self.ctx.status(f"Board image saved ({dpi} dpi): {path}")

    # ============================================================ frames
    def process(self, frame):
        """Runs in the vision thread."""
        det = self.pipeline.detect(frame)
        display = frame
        calib = self.state.calibration
        if self.show_undistorted and calib is not None:
            h, w = frame.shape[:2]
            key = (w, h, id(calib))
            if self._undist_cache[0] != key:
                c = calib.for_resolution(w, h)
                newK, _ = cv2.getOptimalNewCameraMatrix(c.K, c.dist, (w, h), 0.0)
                maps = cv2.initUndistortRectifyMap(c.K, c.dist, None, newK, (w, h), cv2.CV_16SC2)
                self._undist_cache = (key, maps)
            m1, m2 = self._undist_cache[1]
            display = cv2.remap(frame, m1, m2, cv2.INTER_LINEAR)
        return display, det

    def on_packet(self, pkt):
        det = pkt.result if isinstance(pkt.result, CharucoDetection) else None
        self.view.set_image(pkt.image, det if det is not None else False)
        if det is None:
            return
        self.last_det, self.last_frame = det, pkt.frame
        now = time.monotonic()

        motion = self._motion(det, self._prev_det)
        self._prev_det = det
        if motion is not None and motion < 1.2:
            self._still_since = self._still_since or now
        else:
            self._still_since = None
        still = self._still_since is not None and now - self._still_since > 0.25

        if (self.auto.isChecked() and self._job is None and det.count >= self.min_corners.value() and still
                and now - self._last_capture_t > self.interval.value()
                and self.session.novelty(det) > 0.04):
            self.capture()

        if now - self._last_ui > 0.1:
            self._last_ui = now
            self.det_bar.setMaximum(self.board_cfg.total_corners)
            self.det_bar.setValue(min(det.count, self.board_cfg.total_corners))
            ok = det.count >= self.min_corners.value()
            self.det_bar.setStyleSheet("" if ok else f"QProgressBar::chunk {{ background: {theme.WARN}; }}")
            self.legacy_btn.setVisible(det.legacy_hint)
            self.v_sharp.setText(f"{det.sharpness:.0f}" if det.count else "-")
            self.v_motion.setText("-" if motion is None else
                                  f"<span style='color:{theme.ACCENT if still else theme.WARN}'>"
                                  f"{motion:.1f} px{'  (still)' if still else ''}</span>")

    @staticmethod
    def _motion(a: CharucoDetection | None, b: CharucoDetection | None):
        if a is None or b is None or a.count == 0 or b.count == 0:
            return None
        ma, mb = a.id_map(), b.id_map()
        common = set(ma) & set(mb)
        if len(common) < 4:
            return None
        return float(np.mean([np.linalg.norm(ma[i] - mb[i]) for i in common]))

    def capture(self):
        det, frame = self.last_det, self.last_frame
        if det is None or frame is None or det.count < 6:
            self.ctx.status("No board detected - nothing captured.")
            return
        if det.count < self.min_corners.value():
            self.ctx.status(f"Only {det.count} corners visible (min {self.min_corners.value()}).")
            return
        try:
            cap = self.session.add(det, frame)
        except ValueError as exc:
            self.ctx.status(str(exc))
            return
        self._last_capture_t = time.monotonic()
        self._flash_until = time.monotonic() + 0.18
        thumb = cap.thumbnail.copy()
        sx = thumb.shape[1] / det.image_size[0]
        for p in det.points():
            cv2.circle(thumb, (int(p[0] * sx), int(p[1] * sx)), 1, (45, 226, 155), -1)
        cap.thumbnail = thumb
        intr = self.state.intrinsics(*det.image_size)
        cap.estimate_pose(self.pipeline.board, intr.K, intr.dist)
        self._add_gallery_item(len(self.session.captures) - 1)
        self._update_counts()
        if self.left_stack.currentIndex() == 1:
            self._refresh_poses()

    def _add_gallery_item(self, idx):
        cap = self.session.captures[idx]
        item = QListWidgetItem(QIcon(QPixmap.fromImage(bgr_to_qimage(cap.thumbnail))), self._item_text(idx))
        self.gallery.addItem(item)
        self.gallery.scrollToItem(item)

    def _item_text(self, idx):
        cap = self.session.captures[idx]
        t = f"#{idx + 1}  {len(cap.ids)} pts"
        if cap.error is not None:
            t += f"  {cap.error:.2f}px"
        return t

    def _rebuild_gallery(self):
        self.gallery.clear()
        for i in range(len(self.session.captures)):
            self._add_gallery_item(i)
        self._color_gallery()
        self._update_counts()
        self._refresh_poses()

    def _color_gallery(self):
        errs = [c.error for c in self.session.captures if c.error is not None]
        mean = float(np.mean(errs)) if errs else 0
        for i, c in enumerate(self.session.captures):
            it = self.gallery.item(i)
            if it is None:
                continue
            it.setText(self._item_text(i))
            bad = c.error is not None and errs and c.error > max(1.0, 2.0 * mean)
            it.setForeground(QColor(theme.BAD if bad else theme.TEXT))

    def _update_counts(self):
        n = len(self.session.captures)
        self.gallery_title.setText(f"Captures ({n})")
        self.count_bar.setValue(min(n, RECOMMENDED_CAPTURES))
        self.cov_bar.setValue(int(round(self.session.coverage() * 100)))
        self._coverage_pts = self.session.all_points()

    def _delete_selected(self):
        rows = sorted({self.gallery.row(i) for i in self.gallery.selectedItems()}, reverse=True)
        for r in rows:
            self.session.remove(r)
        self._rebuild_gallery()

    def _clear_all(self):
        if self.session.captures and QMessageBox.question(
                self, "Clear captures", f"Delete all {len(self.session.captures)} captures?") \
                != QMessageBox.StandardButton.Yes:
            return
        self.session.clear()
        self._rebuild_gallery()

    # ============================================================ calibrate
    def run_calibration(self):
        if self._job is not None:
            return
        n = len(self.session.captures)
        if n < 4:
            QMessageBox.information(self, "Calibration", f"Capture at least 4 images first "
                                                         f"({RECOMMENDED_CAPTURES} recommended). You have {n}.")
            return
        self.calib_btn.setEnabled(False)
        self.busy.setVisible(True)
        self.ctx.status(f"Calibrating with {n} captures…")
        opts = {"rational_model": bool(self.model.currentData()), "fix_aspect": self.fix_aspect.isChecked()}
        self._job = CalibrateJob(self.session, self.pipeline.board, opts)
        self._job.done.connect(self._calibration_done)
        self._job.start()

    def _calibration_done(self, result):
        job, self._job = self._job, None
        if job is not None:
            job.wait()
        self.calib_btn.setEnabled(True)
        self.busy.setVisible(False)
        if isinstance(result, Exception):
            QMessageBox.warning(self, "Calibration failed", str(result))
            self.ctx.status(f"Calibration failed: {result}")
            return
        calib: CameraCalibration = result
        calib.camera_name = self.state.camera_name
        calib.board = self.board_cfg.__dict__.copy()
        path = default_calibration_path(self.state.camera_name or "camera", calib.width, calib.height)
        try:
            calib.save(path)
        except OSError as exc:
            self.ctx.status(f"Could not save calibration: {exc}")
            path = ""
        self.state.set_calibration(calib, str(path))
        self._color_gallery()
        self._refresh_poses()
        verdict, _ = rms_verdict(calib.rms)
        self.ctx.status(f"Calibration done: RMS {calib.rms:.3f} px ({verdict}) - saved to {path}")

    def _prune_and_recalibrate(self):
        errs = [(c.error, i) for i, c in enumerate(self.session.captures) if c.error is not None]
        if len(errs) < 8:
            self.ctx.status("Need a calibration with 8+ captures first.")
            return
        errs.sort(reverse=True)
        drop = sorted([i for _, i in errs[:max(1, int(len(errs) * 0.15))]], reverse=True)
        for i in drop:
            self.session.remove(i)
        self._rebuild_gallery()
        self.ctx.status(f"Dropped {len(drop)} worst captures - recalibrating…")
        self.run_calibration()

    def _show_result(self):
        c = self.state.calibration
        if c is None:
            self.rms_label.setText("-")
            self.rms_verdict.setText("No calibration yet")
            for v in self.res.values():
                v.setText("-")
            self.dist_label.setText("-")
            self.path_label.setText("")
            return
        verdict, color = rms_verdict(c.rms) if c.rms == c.rms else ("loaded", theme.TEXT)
        self.rms_label.setText(f"{c.rms:.3f}" if c.rms == c.rms else "-")
        self.rms_label.setStyleSheet(f"color: {color};")
        self.rms_verdict.setText(f"<span style='color:{color}'>{verdict}</span>")
        h, v, d = c.fov_deg()
        vals = {"fx": f"{c.fx:.2f}", "fy": f"{c.fy:.2f}", "cx": f"{c.cx:.2f}", "cy": f"{c.cy:.2f}",
                "res": f"{c.width}x{c.height}", "imgs": str(c.num_images),
                "hfov": f"{h:.2f}°", "vfov": f"{v:.2f}°", "dfov": f"{d:.2f}°"}
        a = c.pixel_aspect
        a_ok = 0.99 <= a <= 1.01
        vals["aspect"] = (f"<span style='color:{theme.ACCENT if a_ok else theme.WARN}'>{a:.4f}</span>"
                          + ("" if a_ok else " (check board flatness / size)"))
        for k, val in vals.items():
            self.res[k].setText(val)
        self.dist_label.setText("  ".join(f"{x:.4f}" for x in np.asarray(c.dist).ravel()))
        cp = Path(self.state.calibration_path) if self.state.calibration_path else None
        self.path_label.setText(f"File: {cp.parent.name}/{cp.name}" if cp else "Not saved")
        self.path_label.setToolTip(str(cp) if cp else "")
        self._undist_cache = (None, None)

    def _toggle_undistort(self, on):
        if on and self.state.calibration is None:
            self.ctx.status("Calibrate first to see the undistorted image.")
        self.show_undistorted = on

    def _save_as(self):
        c = self.state.calibration
        if c is None:
            self.ctx.status("Nothing to save yet.")
            return
        default = str(default_calibration_path(self.state.camera_name or "camera", c.width, c.height))
        path, _ = QFileDialog.getSaveFileName(self, "Save calibration", default, "JSON (*.json)")
        if path:
            c.save(path)
            self.state.calibration_path = path
            self._show_result()
            self.ctx.status(f"Calibration saved: {path}")

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load calibration",
                                              str(default_calibration_path("x", 1, 1).parent), "JSON (*.json)")
        if not path:
            return
        try:
            c = CameraCalibration.load(path)
        except Exception as exc:
            QMessageBox.warning(self, "Load calibration", f"Could not read calibration:\n{exc}")
            return
        self.state.set_calibration(c, path)
        w, h = self.state.frame_size
        if (w, h) != (0, 0) and (w, h) != (c.width, c.height):
            self.ctx.status(f"Calibration is for {c.width}x{c.height}, camera runs {w}x{h} - intrinsics are scaled.")
        else:
            self.ctx.status(f"Calibration loaded: {path}")

    # ============================================================ overlay
    def _paint_overlay(self, p: QPainter, view: VideoView):
        det = view.data if isinstance(view.data, CharucoDetection) else None
        r = view.image_rect()
        if self.show_undistorted and self.state.calibration is not None:
            draw_label(p, QPointF(r.x() + 8, r.y() + 28), "UNDISTORTED PREVIEW", fg=theme.INFO,
                       align_center=False, size=10)
        else:
            if self.show_cov.isChecked() and len(self._coverage_pts):
                c = QColor(theme.INFO)
                c.setAlpha(120)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(c)
                step = max(1, len(self._coverage_pts) // 4000)
                for pt in self._coverage_pts[::step]:
                    p.drawEllipse(view.map(pt[0], pt[1]), 2.0, 2.0)
            if det is not None:
                if det.marker_corners is not None:
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.setPen(QPen(QColor(76, 195, 255, 150), 1.2))
                    for mc in det.marker_corners:
                        pts = [view.map(*q) for q in mc.reshape(-1, 2)]
                        for a in range(4):
                            p.drawLine(pts[a], pts[(a + 1) % 4])
                if det.count:
                    pts = det.points()
                    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
                    p.setPen(QPen(QColor(theme.ACCENT), 2))
                    for a in range(len(hull)):
                        p.drawLine(view.map(*hull[a]), view.map(*hull[(a + 1) % len(hull)]))
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QColor(theme.ACCENT))
                    for q in pts:
                        p.drawEllipse(view.map(q[0], q[1]), 3.2, 3.2)
            n = det.count if det is not None else 0
            col = theme.ACCENT if n >= self.min_corners.value() else (theme.WARN if n else theme.BAD)
            draw_label(p, QPointF(r.x() + 8, r.y() + 28),
                       f"BOARD {n}/{self.board_cfg.total_corners} corners   |   captures {len(self.session.captures)}",
                       fg=col, align_center=False, size=10)
        if time.monotonic() < self._flash_until:
            p.setPen(QPen(QColor(255, 255, 255, 220), 8))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(r.adjusted(4, 4, -4, -4))
            QTimer.singleShot(200, view.update)
