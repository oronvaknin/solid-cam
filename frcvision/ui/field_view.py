"""Top-down FRC field map with AprilTags and the estimated robot pose."""

from __future__ import annotations

import math
from collections import deque

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..apriltag import PoseEstimate
from ..field_layout import FieldLayout
from . import theme


class FieldView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout_: FieldLayout | None = None
        self.pose: PoseEstimate | None = None
        self.visible_ids: set[int] = set()
        self.hfov_deg = 70.0
        self.robot_size = (0.8, 0.8)
        self.trail: deque = deque(maxlen=90)
        self._mouse: QPointF | None = None
        self.setMouseTracking(True)
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # -------------------------------------------------------------------- API
    def set_layout(self, layout: FieldLayout | None):
        self.layout_ = layout
        self.trail.clear()
        self.update()

    def set_state(self, pose: PoseEstimate | None, visible_ids):
        self.pose = pose
        self.visible_ids = set(visible_ids)
        if pose is not None:
            T = pose.T_field_robot
            self.trail.append((float(T[0, 3]), float(T[1, 3])))
        self.update()

    def clear_trail(self):
        self.trail.clear()
        self.update()

    # -------------------------------------------------------------- transform
    def _geom(self):
        L = self.layout_.length if self.layout_ else 16.541
        W = self.layout_.width if self.layout_ else 8.069
        m = 26
        s = min((self.width() - 2 * m) / L, (self.height() - 2 * m) / W)
        ox = (self.width() - L * s) / 2
        oy = (self.height() - W * s) / 2
        return L, W, s, ox, oy

    def to_px(self, x, y) -> QPointF:
        L, W, s, ox, oy = self._geom()
        return QPointF(ox + x * s, oy + (W - y) * s)

    def to_field(self, px: QPointF):
        L, W, s, ox, oy = self._geom()
        return (px.x() - ox) / s, W - (px.y() - oy) / s

    # ---------------------------------------------------------------- events
    def mouseMoveEvent(self, ev):
        self._mouse = ev.position()
        self.update()

    def leaveEvent(self, _ev):
        self._mouse = None
        self.update()

    # ---------------------------------------------------------------- paint
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor("#0f1113"))
        L, W, s, ox, oy = self._geom()
        field = QRectF(ox, oy, L * s, W * s)

        # carpet + alliance tint
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#1a1f1c"))
        p.drawRoundedRect(field, 6, 6)
        zone = 4.03 * s  # alliance zone depth (approx.)
        blue = QColor(theme.BLUE_ALLIANCE)
        blue.setAlpha(28)
        red = QColor(theme.RED_ALLIANCE)
        red.setAlpha(28)
        p.setBrush(blue)
        p.drawRect(QRectF(field.left(), field.top(), zone, field.height()))
        p.setBrush(red)
        p.drawRect(QRectF(field.right() - zone, field.top(), zone, field.height()))

        # grid every metre
        p.setPen(QPen(QColor(255, 255, 255, 14), 1))
        for i in range(1, int(L) + 1):
            x = ox + i * s
            p.drawLine(QPointF(x, field.top()), QPointF(x, field.bottom()))
        for j in range(1, int(W) + 1):
            y = oy + j * s
            p.drawLine(QPointF(field.left(), y), QPointF(field.right(), y))

        # outline + centre line + alliance walls
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 120), 1.5))
        p.drawRoundedRect(field, 6, 6)
        p.setPen(QPen(QColor(255, 255, 255, 60), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(ox + L * s / 2, field.top()), QPointF(ox + L * s / 2, field.bottom()))
        p.setPen(QPen(QColor(theme.BLUE_ALLIANCE), 4))
        p.drawLine(field.topLeft(), field.bottomLeft())
        p.setPen(QPen(QColor(theme.RED_ALLIANCE), 4))
        p.drawLine(field.topRight(), field.bottomRight())

        f = QFont(self.font())
        f.setPointSizeF(8)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(theme.BLUE_ALLIANCE))
        p.drawText(QRectF(field.left(), field.bottom() + 4, 120, 18), Qt.AlignmentFlag.AlignLeft, "BLUE  (0,0)")
        p.setPen(QColor(theme.RED_ALLIANCE))
        p.drawText(QRectF(field.right() - 120, field.bottom() + 4, 120, 18), Qt.AlignmentFlag.AlignRight, "RED")
        if self.layout_:
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(field.left(), field.top() - 20, field.width(), 18),
                       Qt.AlignmentFlag.AlignCenter,
                       f"{self.layout_.name}  |  {len(self.layout_.tags)} tags  |  {L:.3f} x {W:.3f} m")

        self._draw_tags(p, s)
        self._draw_robot(p, s)
        self._draw_mouse(p)

    def _draw_tags(self, p: QPainter, s: float):
        if not self.layout_:
            return
        f = QFont(self.font())
        f.setPointSizeF(7.5)
        f.setBold(True)
        p.setFont(f)
        half = max(0.17 * s, 5)
        for tid in sorted(self.layout_.tags):
            x, y, z, yaw = self.layout_.tag_xy_yaw(tid)
            seen = tid in self.visible_ids
            c = self.to_px(x, y)
            # tag face is perpendicular to its normal (yaw)
            nx, ny = math.cos(yaw), -math.sin(yaw)  # screen space (y flipped)
            tx, ty = -ny, nx
            a = QPointF(c.x() + tx * half, c.y() + ty * half)
            b = QPointF(c.x() - tx * half, c.y() - ty * half)
            color = QColor(theme.ACCENT) if seen else QColor("#c9ced6")
            p.setPen(QPen(color, 4 if seen else 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(a, b)
            tip = QPointF(c.x() + nx * half * 1.3, c.y() + ny * half * 1.3)
            p.setPen(QPen(color, 1.5))
            p.drawLine(c, tip)
            # label offset along the normal
            lp = QPointF(c.x() + nx * (half * 2.6), c.y() + ny * (half * 2.6))
            r = QRectF(lp.x() - 11, lp.y() - 8, 22, 16)
            if seen:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(theme.ACCENT))
                p.drawRoundedRect(r, 4, 4)
                p.setPen(QColor("#0b0c0e"))
            else:
                p.setPen(QColor("#9aa3ae"))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, str(tid))
            p.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_robot(self, p: QPainter, s: float):
        # trail
        if len(self.trail) > 1:
            path = QPainterPath(self.to_px(*self.trail[0]))
            for pt in list(self.trail)[1:]:
                path.lineTo(self.to_px(*pt))
            p.setPen(QPen(QColor(155, 226, 45, 90), 1.5))
            p.drawPath(path)
        if self.pose is None:
            return
        Tc = self.pose.T_field_camera
        Tr = self.pose.T_field_robot
        cam = self.to_px(Tc[0, 3], Tc[1, 3])
        cam_yaw = math.atan2(Tc[1, 0], Tc[0, 0])

        # FOV wedge
        rng = 4.5
        wedge = QPolygonF([cam])
        for k in range(-10, 11):
            a = cam_yaw + math.radians(self.hfov_deg / 2) * k / 10
            wedge.append(self.to_px(Tc[0, 3] + rng * math.cos(a), Tc[1, 3] + rng * math.sin(a)))
        fov = QColor(theme.ACCENT)
        fov.setAlpha(22)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(fov)
        p.drawPolygon(wedge)

        # lines to tags used in the estimate
        if self.layout_:
            p.setPen(QPen(QColor(155, 226, 45, 160), 1, Qt.PenStyle.DashLine))
            for tid in self.pose.tag_ids:
                if tid in self.layout_.tags:
                    x, y, _, _ = self.layout_.tag_xy_yaw(tid)
                    p.drawLine(cam, self.to_px(x, y))

        # robot body
        rx, ry = Tr[0, 3], Tr[1, 3]
        yaw = math.atan2(Tr[1, 0], Tr[0, 0])
        lx, ly = self.robot_size[0] / 2, self.robot_size[1] / 2
        pts = []
        for dx, dy in [(lx, ly), (lx, -ly), (-lx, -ly), (-lx, ly)]:
            pts.append(self.to_px(rx + dx * math.cos(yaw) - dy * math.sin(yaw),
                                  ry + dx * math.sin(yaw) + dy * math.cos(yaw)))
        body = QColor(theme.ACCENT)
        body.setAlpha(60)
        p.setBrush(body)
        p.setPen(QPen(QColor(theme.ACCENT), 2))
        p.drawPolygon(QPolygonF(pts))
        # heading arrow
        c = self.to_px(rx, ry)
        tip = self.to_px(rx + lx * 1.4 * math.cos(yaw), ry + lx * 1.4 * math.sin(yaw))
        p.setPen(QPen(QColor(theme.ACCENT), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(c, tip)
        # camera dot
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.INFO))
        p.drawEllipse(cam, 4, 4)

        f = QFont(self.font())
        f.setPointSizeF(8.5)
        f.setBold(True)
        p.setFont(f)
        txt = f"x {rx:.2f}  y {ry:.2f}  {math.degrees(yaw):.1f}°"
        fm = p.fontMetrics()
        r = QRectF(c.x() - fm.horizontalAdvance(txt) / 2 - 6, c.y() + ly * s + 8, fm.horizontalAdvance(txt) + 12, 18)
        p.setBrush(QColor(11, 12, 14, 210))
        p.drawRoundedRect(r, 4, 4)
        p.setPen(QColor(theme.ACCENT))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, txt)

    def _draw_mouse(self, p: QPainter):
        if self._mouse is None:
            return
        x, y = self.to_field(self._mouse)
        L, W, *_ = self._geom()
        if not (0 <= x <= L and 0 <= y <= W):
            return
        f = QFont(self.font())
        f.setPointSizeF(8)
        p.setFont(f)
        p.setPen(QColor(theme.MUTED))
        p.drawText(QRectF(self.width() - 170, self.height() - 22, 160, 18),
                   Qt.AlignmentFlag.AlignRight, f"({x:.2f}, {y:.2f}) m")
