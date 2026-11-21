"""A small QPainter 3D renderer (painter's algorithm) with an orbit camera.

No OpenGL needed - it works everywhere Qt works (also headless / remote desktop).
World axes follow WPILib: +X forward, +Y left, +Z up.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPen, QPolygonF, QTransform
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QWidget

NEAR = 0.05
LIGHT = np.array([0.35, 0.55, 1.0]) / np.linalg.norm([0.35, 0.55, 1.0])


@dataclass
class Prim:
    kind: str  # "poly" | "line" | "image" | "text" | "point"
    pts: np.ndarray  # (N,3) world points
    color: QColor | None = None
    edge: QColor | None = None
    width: float = 1.0
    layer: int = 1  # 0 = ground (insertion order), 1 = depth-sorted, 2 = overlay (always on top)
    image: QImage | None = None
    text: str = ""
    bias: float = 0.0  # depth bias toward the viewer (draw later)
    shade: bool = True
    dash: bool = False
    size: float = 8.0
    bold: bool = True


def poly(pts, color, edge=None, width=1.0, **kw) -> Prim:
    return Prim("poly", np.asarray(pts, float), QColor(color), QColor(edge) if edge else None, width, **kw)


def line(pts, color, width=1.0, **kw) -> Prim:
    return Prim("line", np.asarray(pts, float), QColor(color), None, width, **kw)


def text(pt, s, color, size=8.0, **kw) -> Prim:
    return Prim("text", np.asarray([pt], float), QColor(color), text=s, size=size, layer=kw.pop("layer", 2), **kw)


def box(T: np.ndarray, sx, sy, sz, z0, color, edge=None, top_color=None, alpha=255) -> list[Prim]:
    """Axis-aligned box in the frame T (centre at x=y=0, from z0 to z0+sz)."""
    hx, hy = sx / 2, sy / 2
    c = np.array([[x, y, z, 1.0] for z in (z0, z0 + sz) for y in (-hy, hy) for x in (-hx, hx)])
    w = (T @ c.T).T[:, :3]
    faces = [(4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]  # top + 4 sides
    out = []
    for n, f in enumerate(faces):
        col = QColor(top_color if (n == 0 and top_color) else color)
        col.setAlpha(alpha)
        out.append(poly(w[list(f)], col, edge, 1.0))
    return out


class View3D(QWidget):
    """Base widget: subclasses implement build_scene() -> list[Prim]."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.target = np.array([0.0, 0.0, 0.0])
        self.yaw = math.radians(180)
        self.pitch = math.radians(35)
        self.dist = 10.0
        self.fov = math.radians(50)
        self.home = None
        self._drag = None
        self._anim = None
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)
        self._anim_timer.timeout.connect(self._step_anim)
        self.bg_top = QColor("#141a20")
        self.bg_bottom = QColor("#0b0d10")
        self.setMouseTracking(True)
        self.setMinimumSize(300, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.toolbar = QWidget(self)
        self._tb = QHBoxLayout(self.toolbar)
        self._tb.setContentsMargins(0, 0, 0, 0)
        self._tb.setSpacing(4)

    # ------------------------------------------------------------- camera
    def eye(self):
        cp = math.cos(self.pitch)
        return self.target + self.dist * np.array([cp * math.cos(self.yaw), cp * math.sin(self.yaw), math.sin(self.pitch)])

    def basis(self):
        eye = self.eye()
        f = self.target - eye
        f /= np.linalg.norm(f)
        r = np.cross(f, [0.0, 0.0, 1.0])
        if np.linalg.norm(r) < 1e-6:
            r = np.array([math.sin(self.yaw), -math.cos(self.yaw), 0.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        return eye, r, u, f

    def fit_distance(self, pts, target, yaw, pitch, margin=0.08) -> float:
        """Smallest orbit distance at which every point in `pts` is on screen."""
        saved = (self.target, self.yaw, self.pitch, self.dist)
        self.target, self.yaw, self.pitch = np.asarray(target, float), yaw, pitch
        w, h = max(self.width(), 80), max(self.height(), 80)
        F = (h / 2) / math.tan(self.fov / 2)
        pts = np.asarray(pts, float)
        lo, hi = 0.3, 300.0
        for _ in range(32):
            self.dist = (lo + hi) / 2
            eye, r, u, f = self.basis()
            d = pts - eye
            x, y, z = d @ r, d @ u, d @ f
            ok = bool((z > NEAR).all() and (np.abs(F * x / z) <= w / 2 * (1 - margin)).all()
                      and (np.abs(F * y / z) <= h / 2 * (1 - margin) - 14).all())
            if ok:
                hi = self.dist
            else:
                lo = self.dist
        self.target, self.yaw, self.pitch, self.dist = saved
        return hi

    def set_view(self, target=None, yaw=None, pitch=None, dist=None, animate=True):
        end = (
            np.array(target if target is not None else self.target, float),
            self.yaw if yaw is None else yaw,
            self.pitch if pitch is None else pitch,
            self.dist if dist is None else dist,
        )
        if not animate or not self.isVisible():
            self.target, self.yaw, self.pitch, self.dist = end[0], end[1], end[2], end[3]
            self.update()
            return
        # shortest yaw path
        dy = (end[1] - self.yaw + math.pi) % (2 * math.pi) - math.pi
        self._anim = (time.monotonic(), (self.target.copy(), self.yaw, self.pitch, self.dist),
                      (end[0], self.yaw + dy, end[2], end[3]))
        self._anim_timer.start()

    def _step_anim(self):
        if self._anim is None:
            self._anim_timer.stop()
            return
        t0, a, b = self._anim
        t = min(1.0, (time.monotonic() - t0) / 0.45)
        e = 1 - (1 - t) ** 3
        self.target = a[0] + (b[0] - a[0]) * e
        self.yaw = a[1] + (b[1] - a[1]) * e
        self.pitch = a[2] + (b[2] - a[2]) * e
        self.dist = a[3] + (b[3] - a[3]) * e
        if t >= 1.0:
            self._anim = None
            self._anim_timer.stop()
        self.update()

    def add_tool_button(self, text, slot, checkable=False, tip=""):
        b = QPushButton(text, self.toolbar)
        b.setObjectName("Overlay")
        b.setCheckable(checkable)
        b.setToolTip(tip)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.clicked.connect(slot)
        self._tb.addWidget(b)
        self.toolbar.adjustSize()
        return b

    def resizeEvent(self, ev):
        self.toolbar.adjustSize()
        self.toolbar.move(self.width() - self.toolbar.width() - 8, 8)
        super().resizeEvent(ev)

    # -------------------------------------------------------------- input
    def mousePressEvent(self, ev):
        self._drag = (ev.position(), ev.buttons(), self.yaw, self.pitch, self.target.copy())
        self._anim = None

    def mouseReleaseEvent(self, _ev):
        self._drag = None

    def mouseMoveEvent(self, ev):
        if not self._drag:
            return
        p0, buttons, yaw0, pitch0, target0 = self._drag
        d = ev.position() - p0
        if buttons & Qt.MouseButton.LeftButton:
            self.yaw = yaw0 - d.x() * 0.006
            self.pitch = max(math.radians(3), min(math.radians(89), pitch0 + d.y() * 0.006))
        else:
            _, r, u, f = self.basis()
            fwd = np.array([f[0], f[1], 0.0])
            if np.linalg.norm(fwd) > 1e-6:
                fwd /= np.linalg.norm(fwd)
            k = self.dist * 0.0016
            self.target = target0 - r * d.x() * k + fwd * d.y() * k
            self.on_user_pan()
        self.update()

    def on_user_pan(self):
        pass

    def wheelEvent(self, ev):
        steps = ev.angleDelta().y() / 120.0
        self.dist = max(0.6, min(80.0, self.dist * (0.88 ** steps)))
        self._anim = None
        self.update()

    def mouseDoubleClickEvent(self, _ev):
        if self.home:
            self.set_view(*self.home)

    # -------------------------------------------------------------- scene
    def build_scene(self) -> list[Prim]:
        return []

    def paint_overlay(self, p: QPainter):
        pass

    # ------------------------------------------------------------ render
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        g = QLinearGradient(0, 0, 0, self.height())
        g.setColorAt(0, self.bg_top)
        g.setColorAt(1, self.bg_bottom)
        p.fillRect(self.rect(), g)

        eye, r, u, f = self.basis()
        w, h = self.width(), self.height()
        F = (h / 2) / math.tan(self.fov / 2)
        cx, cy = w / 2, h / 2

        def to_cam(P):
            d = P - eye
            return np.stack([d @ r, d @ u, d @ f], axis=-1)

        def proj(cp):
            return [QPointF(cx + F * x / z, cy - F * y / z) for x, y, z in cp]

        items = []
        for order, prim in enumerate(self.build_scene()):
            cp = to_cam(prim.pts)
            if prim.kind in ("poly", "image"):
                if prim.kind == "image" and (cp[:, 2] < NEAR).any():
                    continue
                if prim.kind == "poly":
                    cp = _clip_poly(cp)
                    if cp is None:
                        continue
                depth = float(cp[:, 2].mean()) - prim.bias
                items.append((prim.layer, depth, order, prim, cp))
            elif prim.kind == "line":
                for a, b in zip(cp[:-1], cp[1:]):
                    seg = _clip_seg(a, b)
                    if seg is not None:
                        items.append((prim.layer, float((seg[0][2] + seg[1][2]) / 2) - prim.bias, order, prim, seg))
            else:  # text / point
                if cp[0, 2] > NEAR:
                    items.append((prim.layer, float(cp[0, 2]) - prim.bias, order, prim, cp))

        items.sort(key=lambda it: (it[0], -it[1] if it[0] == 1 else it[2]))
        font = QFont(self.font())
        for _layer, depth, _o, prim, cp in items:
            if prim.kind == "poly":
                col = QColor(prim.color)
                if prim.shade and len(prim.pts) >= 3:
                    n = np.cross(prim.pts[1] - prim.pts[0], prim.pts[2] - prim.pts[0])
                    nn = np.linalg.norm(n)
                    if nn > 1e-9:
                        k = 0.62 + 0.38 * abs(float(n @ LIGHT) / nn)
                        col = QColor(int(col.red() * k), int(col.green() * k), int(col.blue() * k), col.alpha())
                p.setBrush(col)
                if prim.edge is not None:
                    p.setPen(QPen(prim.edge, prim.width))
                else:
                    p.setPen(Qt.PenStyle.NoPen)
                p.drawPolygon(QPolygonF(proj(cp)))
            elif prim.kind == "image":
                dst = proj(cp)
                iw, ih = prim.image.width(), prim.image.height()
                src = QPolygonF([QPointF(0, 0), QPointF(iw, 0), QPointF(iw, ih), QPointF(0, ih)])
                t = QTransform()
                if QTransform.quadToQuad(src, QPolygonF(dst), t):
                    p.save()
                    p.setTransform(t, True)
                    p.drawImage(QPointF(0, 0), prim.image)
                    p.restore()
            elif prim.kind == "line":
                pen = QPen(prim.color, prim.width)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                if prim.dash:
                    pen.setStyle(Qt.PenStyle.DashLine)
                p.setPen(pen)
                a, b = proj(cp)
                p.drawLine(a, b)
            elif prim.kind == "point":
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(prim.color)
                q = proj(cp)[0]
                p.drawEllipse(q, prim.size, prim.size)
            elif prim.kind == "text":
                q = proj(cp)[0]
                font.setPointSizeF(prim.size)
                font.setBold(prim.bold)
                p.setFont(font)
                fm = p.fontMetrics()
                tw = fm.horizontalAdvance(prim.text) + 8
                rect = QRectF(q.x() - tw / 2, q.y() - fm.height() / 2 - 1, tw, fm.height() + 2)
                if prim.edge is not None:
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(prim.edge)
                    p.drawRoundedRect(rect, 4, 4)
                p.setPen(prim.color)
                p.drawText(rect, Qt.AlignmentFlag.AlignCenter, prim.text)
        p.resetTransform()
        self.paint_overlay(p)


def _clip_poly(cp: np.ndarray):
    if (cp[:, 2] >= NEAR).all():
        return cp
    out = []
    n = len(cp)
    for i in range(n):
        a, b = cp[i], cp[(i + 1) % n]
        ain, bin_ = a[2] >= NEAR, b[2] >= NEAR
        if ain:
            out.append(a)
        if ain != bin_:
            t = (NEAR - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return np.array(out) if len(out) >= 3 else None


def _clip_seg(a, b):
    ain, bin_ = a[2] >= NEAR, b[2] >= NEAR
    if ain and bin_:
        return a, b
    if not ain and not bin_:
        return None
    t = (NEAR - a[2]) / (b[2] - a[2])
    m = a + t * (b - a)
    return (a, m) if ain else (m, b)
