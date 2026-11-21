"""3D FRC field: carpet, walls, every AprilTag (real 36h11 texture), robots, camera frustums."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter

from ..field_layout import FieldLayout
from . import theme
from .view3d import Prim, View3D, box, line, poly, text

_TAG_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
_TAG_CACHE: dict[int, QImage] = {}


def tag_image(tag_id: int) -> QImage:
    """Official-orientation tag36h11 image incl. the white border (10x10 cells)."""
    img = _TAG_CACHE.get(tag_id)
    if img is None:
        cell = 12
        try:
            m = cv2.aruco.generateImageMarker(_TAG_DICT, tag_id, 8 * cell)
            m = np.rot90(m, 2).copy()  # OpenCV's bit pattern is the official tag rotated 180 deg
        except cv2.error:
            m = np.full((8 * cell, 8 * cell), 128, np.uint8)
        full = np.full((10 * cell, 10 * cell), 255, np.uint8)
        full[cell:9 * cell, cell:9 * cell] = m
        img = QImage(full.data, full.shape[1], full.shape[0], full.strides[0],
                     QImage.Format.Format_Grayscale8).copy()
        _TAG_CACHE[tag_id] = img
    return img


@dataclass
class RobotView:
    T_robot: np.ndarray
    T_cam: np.ndarray | None = None
    tag_ids: list = field(default_factory=list)
    color: str = theme.ACCENT
    label: str = ""
    hfov: float = 70.0
    vfov: float = 45.0


class Field3DView(View3D):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout_: FieldLayout | None = None
        self.robots: dict[str, RobotView] = {}
        self.visible_ids: set[int] = set()
        self.trails: dict[str, deque] = {}
        self.robot_size = (0.8, 0.8)
        self.alliance = "blue"
        self.follow_key: str | None = None
        self._static: list[Prim] = []
        self.L, self.W = 16.541, 8.069
        self._reset_home()
        b = self.add_tool_button("Blue", lambda: self.preset("blue"), tip="Blue driver-station view")
        b.setStyleSheet(f"color: {theme.BLUE_ALLIANCE};")
        r = self.add_tool_button("Red", lambda: self.preset("red"), tip="Red driver-station view")
        r.setStyleSheet(f"color: {theme.RED_ALLIANCE};")
        self.add_tool_button("Top", lambda: self.preset("top"), tip="Top-down view")
        self.add_tool_button("Side", lambda: self.preset("side"), tip="Side view")
        self.follow_btn = self.add_tool_button("Follow", self._toggle_follow, checkable=True,
                                               tip="Camera follows the robot")
        self.setToolTip("Drag: rotate  |  Right-drag: pan  |  Wheel: zoom  |  Double-click: reset")

    # ------------------------------------------------------------------ API
    def set_layout(self, layout: FieldLayout | None):
        self.layout_ = layout
        if layout:
            self.L, self.W = layout.length, layout.width
        self._static = self._build_static()
        for t in self.trails.values():
            t.clear()
        self._reset_home()
        self.preset("home", animate=False)

    def set_robot(self, key: str, robot: RobotView | None):
        if robot is None:
            self.robots.pop(key, None)
        else:
            self.robots[key] = robot
            trail = self.trails.setdefault(key, deque(maxlen=120))
            pt = (float(robot.T_robot[0, 3]), float(robot.T_robot[1, 3]))
            if not trail or math.dist(trail[-1], pt) > 0.02:
                trail.append(pt)
            if self.follow_key == key and self._drag is None and self._anim is None:
                tgt = np.array([pt[0], pt[1], 0.3])
                self.target = self.target + (tgt - self.target) * 0.25
        self.update()

    def set_visible(self, ids):
        self.visible_ids = set(ids)

    def clear_trails(self):
        for t in self.trails.values():
            t.clear()
        self.update()

    PRESETS = {
        "home": (-112, 40),
        "blue": (180, 30),
        "red": (0, 30),
        "top": (-90, 89),
        "side": (-90, 24),
    }

    def _field_points(self):
        L, W = self.L, self.W
        return [[0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0], [0, 0, 2.0], [0, W, 2.0], [L, 0, 2.0], [L, W, 2.0]]

    def preset(self, name: str, animate: bool = True):
        yaw_d, pitch_d = self.PRESETS[name]
        c = np.array([self.L / 2, self.W / 2, 0.4])
        yaw, pitch = math.radians(yaw_d), math.radians(pitch_d)
        dist = self.fit_distance(self._field_points(), c, yaw, pitch)
        self._user_moved = name != "home"
        self.set_view(c, yaw, pitch, dist, animate=animate)

    def _reset_home(self):
        self._user_moved = False
        self.home = (np.array([self.L / 2, self.W / 2, 0.4]), math.radians(-112), math.radians(40), self.L * 0.95)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if not getattr(self, "_user_moved", False) and not self.follow_key:
            self.preset("home", animate=False)

    def mousePressEvent(self, ev):
        self._user_moved = True
        super().mousePressEvent(ev)

    def wheelEvent(self, ev):
        self._user_moved = True
        super().wheelEvent(ev)

    def mouseDoubleClickEvent(self, _ev):
        self.preset("home")

    def _toggle_follow(self, on):
        self.follow_key = next(iter(self.robots), "app") if on else None
        if on:
            self.dist = min(self.dist, 7.0)
            self.update()

    def on_user_pan(self):
        if self.follow_key:
            self.follow_key = None
            self.follow_btn.setChecked(False)

    # --------------------------------------------------------------- static
    def _build_static(self) -> list[Prim]:
        L, W = self.L, self.W
        out: list[Prim] = []
        # surroundings + carpet
        out.append(poly([[-4, -3, -0.01], [L + 4, -3, -0.01], [L + 4, W + 3, -0.01], [-4, W + 3, -0.01]],
                        "#101418", layer=0, shade=False))
        out.append(poly([[0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0]], "#1d2523", layer=0, shade=False))
        zone = 4.03
        out.append(poly([[0, 0, 0], [zone, 0, 0], [zone, W, 0], [0, W, 0]], QColor(47, 123, 255, 34),
                        layer=0, shade=False))
        out.append(poly([[L - zone, 0, 0], [L, 0, 0], [L, W, 0], [L - zone, W, 0]], QColor(255, 64, 64, 34),
                        layer=0, shade=False))
        grid = QColor(255, 255, 255, 16)
        for i in range(1, int(L) + 1):
            out.append(line([[i, 0, 0], [i, W, 0]], grid, 1, layer=0))
        for j in range(1, int(W) + 1):
            out.append(line([[0, j, 0], [L, j, 0]], grid, 1, layer=0))
        out.append(line([[L / 2, 0, 0], [L / 2, W, 0]], QColor(255, 255, 255, 110), 2, layer=0))
        out.append(line([[zone, 0, 0], [zone, W, 0]], QColor(47, 123, 255, 140), 2, layer=0))
        out.append(line([[L - zone, 0, 0], [L - zone, W, 0]], QColor(255, 64, 64, 140), 2, layer=0))
        out.append(line([[0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0], [0, 0, 0]], QColor(255, 255, 255, 150), 2,
                        layer=0))

        # side rails (polycarbonate, 0.5 m) in 1 m panels so depth sorting stays correct
        rail = QColor(200, 220, 255, 20)
        edge = QColor(200, 220, 255, 70)
        for y in (0.0, W):
            xs = np.linspace(0, L, int(math.ceil(L)) + 1)
            for a, b in zip(xs[:-1], xs[1:]):
                out.append(poly([[a, y, 0], [b, y, 0], [b, y, 0.5], [a, y, 0.5]], rail, shade=False))
            out.append(line([[0, y, 0.5], [L, y, 0.5]], edge, 1.5))
        # alliance walls (2 m, tinted)
        for x, col in ((0.0, QColor(47, 123, 255, 46)), (L, QColor(255, 64, 64, 46))):
            ys = np.linspace(0, W, int(math.ceil(W)) + 1)
            for a, b in zip(ys[:-1], ys[1:]):
                out.append(poly([[x, a, 0], [x, b, 0], [x, b, 2.0], [x, a, 2.0]], col, shade=False))
            ec = QColor(col)
            ec.setAlpha(170)
            out.append(line([[x, 0, 2.0], [x, W, 2.0]], ec, 2))
            out.append(line([[x, 0, 0], [x, 0, 2.0]], ec, 1.5))
            out.append(line([[x, W, 0], [x, W, 2.0]], ec, 1.5))

        if self.layout_:
            s = self.layout_.tag_size
            for tid, T in self.layout_.tags.items():
                out.extend(self._tag_prims(tid, T, s))
        return out

    def _tag_prims(self, tid: int, T: np.ndarray, s: float) -> list[Prim]:
        out = []
        n = T[:3, 0]  # out of the tag face
        yv = T[:3, 1]  # viewer's right
        zv = T[:3, 2]
        c = T[:3, 3]
        # mounting panel from the floor to just above the tag
        pw, top = 0.42, c[2] + 0.22
        back = c - n * 0.03
        b0 = back - zv * c[2]
        panel = [b0 - yv * pw / 2, b0 + yv * pw / 2, b0 + yv * pw / 2 + zv * (top), b0 - yv * pw / 2 + zv * (top)]
        out.append(poly(panel, QColor(70, 78, 90, 215), QColor(120, 130, 145, 160), 1))
        # tag (with white border = 10/8 of the black square), drawn as an image on the front face
        h = s * 10 / 8 / 2
        face = c + n * 0.004
        quad = [face - yv * h + zv * h, face + yv * h + zv * h, face + yv * h - zv * h, face - yv * h - zv * h]
        out.append(Prim("image", np.array(quad), image=tag_image(tid), bias=0.02))
        out.append(text(c + zv * (h + 0.16), str(tid), "#c9ced6", 7.5, edge=QColor(11, 12, 14, 170)))
        return out

    # -------------------------------------------------------------- dynamic
    def build_scene(self) -> list[Prim]:
        prims = list(self._static)
        if self.layout_:
            s = self.layout_.tag_size
            for tid in self.visible_ids:
                T = self.layout_.tags.get(tid)
                if T is None:
                    continue
                n, yv, zv, c = T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] + T[:3, 0] * 0.01
                h = s * 10 / 8 / 2 + 0.03
                ring = [c - yv * h + zv * h, c + yv * h + zv * h, c + yv * h - zv * h, c - yv * h - zv * h,
                        c - yv * h + zv * h]
                prims.append(line(ring, theme.ACCENT, 3, bias=0.05))
                prims.append(text(c + zv * (h + 0.14), str(tid), "#0b0c0e", 8, edge=QColor(theme.ACCENT)))
        for key, trail in self.trails.items():
            if len(trail) > 1 and key in self.robots:
                col = QColor(self.robots[key].color)
                col.setAlpha(110)
                prims.append(line([[x, y, 0.02] for x, y in trail], col, 2, layer=0))
        for rv in self.robots.values():
            prims.extend(self._robot_prims(rv))
        return prims

    def _robot_prims(self, rv: RobotView) -> list[Prim]:
        out = []
        T = rv.T_robot.copy()
        # keep the robot drawn upright on the carpet
        yaw = math.atan2(T[1, 0], T[0, 0])
        Tg = np.eye(4)
        Tg[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
        Tg[:2, 3] = T[:2, 3]
        lx, ly = self.robot_size
        col = QColor(rv.color)
        bumper = QColor(theme.BLUE_ALLIANCE if self.alliance == "blue" else theme.RED_ALLIANCE)
        out += box(Tg, lx, ly, 0.13, 0.02, bumper, QColor(0, 0, 0, 120))
        out += box(Tg, lx - 0.16, ly - 0.16, 0.22, 0.15, "#2b3036", QColor(col.red(), col.green(), col.blue(), 150),
                   top_color="#353b42")
        tip = Tg @ np.array([lx / 2 + 0.25, 0, 0.38, 1])
        base = Tg @ np.array([-lx / 4, 0, 0.38, 1])
        out.append(line([base[:3], tip[:3]], col, 4, bias=0.2))
        out.append(text(Tg[:3, 3] + [0, 0, 0.75], rv.label or "robot", "#0b0c0e", 8.5, edge=col))

        if rv.T_cam is not None:
            C = rv.T_cam
            o = C[:3, 3]
            out.append(Prim("point", np.array([o]), QColor(theme.INFO), size=4, layer=1, bias=0.3))
            d = 1.1
            th, tv = math.tan(math.radians(rv.hfov / 2)), math.tan(math.radians(rv.vfov / 2))
            corners = [C @ np.array([d, sy * th * d, sz * tv * d, 1.0]) for sy, sz in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
            fr = QColor(col)
            fr.setAlpha(150)
            for k in corners:
                out.append(line([o, k[:3]], fr, 1.2))
            out.append(line([k[:3] for k in corners] + [corners[0][:3]], fr, 1.2))
            fill = QColor(col)
            fill.setAlpha(28)
            out.append(poly([k[:3] for k in corners], fill, shade=False))
            if self.layout_:
                for tid in rv.tag_ids:
                    Tt = self.layout_.tags.get(tid)
                    if Tt is not None:
                        out.append(line([o, Tt[:3, 3]], QColor(155, 226, 45, 190), 1.6, dash=True, bias=0.1))
        return out

    def paint_overlay(self, p: QPainter):
        f = QFont(self.font())
        f.setPointSizeF(8.5)
        p.setFont(f)
        p.setPen(QColor(theme.MUTED))
        name = self.layout_.name if self.layout_ else "No field map"
        p.drawText(QRectF(10, 8, 600, 18), Qt.AlignmentFlag.AlignLeft, f"{name}   |   3D field")
        y = self.height() - 10
        for key, rv in self.robots.items():
            T = rv.T_robot
            yaw = math.degrees(math.atan2(T[1, 0], T[0, 0]))
            p.setPen(QColor(rv.color))
            f.setBold(True)
            p.setFont(f)
            p.drawText(QRectF(10, y - 18, 520, 18), Qt.AlignmentFlag.AlignLeft,
                       f"{rv.label or key}:  x {T[0, 3]:.2f}   y {T[1, 3]:.2f}   z {T[2, 3]:.2f}   yaw {yaw:.1f}°")
            y -= 18
