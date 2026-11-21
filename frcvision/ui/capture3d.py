"""3D view of where the calibration board was for every capture (like Limelight's 3D point cloud)."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter

from ..geometry import CV_FROM_WPI
from . import theme
from .view3d import Prim, View3D, box, line, poly, text


def error_color(err: float | None, mean: float) -> QColor:
    if err is None:
        return QColor(theme.INFO)
    t = max(0.0, min(1.0, (err / max(mean, 1e-6) - 0.6) / 1.4))  # ~mean -> green, 2x mean -> red
    a, b = QColor(theme.ACCENT), QColor(theme.BAD)
    return QColor(int(a.red() + (b.red() - a.red()) * t), int(a.green() + (b.green() - a.green()) * t),
                  int(a.blue() + (b.blue() - a.blue()) * t))


class CapturePosesView(View3D):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.captures = []
        self.board_rect = None  # (x0, y0, x1, y1) in board units (m)
        self.hfov, self.vfov = 70.0, 45.0
        self.home = (np.array([0.55, 0.0, 0.0]), math.radians(205), math.radians(24), 2.3)
        self.set_view(*self.home, animate=False)
        self.add_tool_button("Behind", lambda: self.set_view(*self.home), tip="View from behind the camera")
        self.add_tool_button("Top", lambda: self.set_view(np.array([0.6, 0, 0]), math.radians(180),
                                                         math.radians(88), 2.6), tip="Top view")
        self.add_tool_button("Side", lambda: self.set_view(np.array([0.6, 0, 0]), math.radians(-90),
                                                          math.radians(5), 2.4), tip="Side view")
        self.setToolTip("Each rectangle is the board in one capture. Colour = per-image error after calibrating.")

    def set_data(self, captures, board, hfov, vfov):
        self.captures = captures
        self.hfov, self.vfov = hfov, vfov
        corners = np.asarray(board.getChessboardCorners()).reshape(-1, 3)
        sq = board.getSquareLength()
        lo, hi = corners.min(axis=0) - sq, corners.max(axis=0) + sq
        self.board_rect = (lo[0], lo[1], hi[0], hi[1])
        self.update()

    def build_scene(self) -> list[Prim]:
        out: list[Prim] = []
        # ground reference + axes at the camera
        g = QColor(255, 255, 255, 14)
        for i in range(-4, 5):
            out.append(line([[0, i * 0.25, -0.6], [2.0, i * 0.25, -0.6]], g, 1, layer=0))
        for i in range(0, 9):
            out.append(line([[i * 0.25, -1.0, -0.6], [i * 0.25, 1.0, -0.6]], g, 1, layer=0))
        out += box(np.eye(4), 0.08, 0.10, 0.07, -0.035, "#3a4048", QColor(theme.INFO))
        d = 0.35
        th, tv = math.tan(math.radians(self.hfov / 2)), math.tan(math.radians(self.vfov / 2))
        far = [np.array([d, sy * th * d, sz * tv * d]) for sy, sz in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        fr = QColor(theme.INFO)
        fr.setAlpha(170)
        for k in far:
            out.append(line([[0, 0, 0], k], fr, 1.2))
        out.append(line(far + [far[0]], fr, 1.2))
        out.append(text([0, 0, 0.12], "camera", "#0b0c0e", 8, edge=QColor(theme.INFO)))
        if self.board_rect is None:
            return out
        x0, y0, x1, y1 = self.board_rect
        local = np.array([[x0, y0, 0, 1], [x1, y0, 0, 1], [x1, y1, 0, 1], [x0, y1, 0, 1]], float).T
        errs = [c.error for c in self.captures if c.error is not None]
        mean = float(np.mean(errs)) if errs else 0.0
        for i, cap in enumerate(self.captures):
            T = getattr(cap, "T_cam_board", None)
            if T is None:
                continue
            p_cv = (T @ local)[:3].T
            p_w = (CV_FROM_WPI.T @ p_cv.T).T
            col = error_color(cap.error, mean)
            fill = QColor(col)
            fill.setAlpha(70)
            out.append(poly(p_w, fill, col, 1.4, shade=False))
            c = p_w.mean(axis=0)
            label = f"#{i + 1}" + (f" {cap.error:.2f}" if cap.error is not None else "")
            out.append(text(c, label, "#e7eaee", 7, edge=QColor(11, 12, 14, 170)))
        return out

    def paint_overlay(self, p: QPainter):
        f = QFont(self.font())
        f.setPointSizeF(8.5)
        p.setFont(f)
        p.setPen(QColor(theme.MUTED))
        n = sum(1 for c in self.captures if getattr(c, "T_cam_board", None) is not None)
        p.drawText(QRectF(10, 8, 600, 18), Qt.AlignmentFlag.AlignLeft,
                   f"{n} board poses  |  aim for variety: near / far, left / right, tilted ~30-45°")
