"""Small vector icons drawn with QPainter (no image files needed)."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

SIZE = 48


def _icon(draw, normal: str, active: str) -> QIcon:
    icon = QIcon()
    for color, mode, state in ((normal, QIcon.Mode.Normal, QIcon.State.Off),
                               (active, QIcon.Mode.Normal, QIcon.State.On),
                               (active, QIcon.Mode.Active, QIcon.State.Off)):
        pm = QPixmap(SIZE, SIZE)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(color), 3.2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        draw(p, QColor(color))
        p.end()
        icon.addPixmap(pm, mode, state)
    return icon


def _camera(p: QPainter, c: QColor):
    p.drawRoundedRect(QRectF(6, 14, 36, 24), 5, 5)
    p.drawEllipse(QPointF(24, 26), 7, 7)
    p.drawLine(QPointF(15, 14), QPointF(19, 9))
    p.drawLine(QPointF(19, 9), QPointF(29, 9))
    p.drawLine(QPointF(29, 9), QPointF(33, 14))


def _tag(p: QPainter, c: QColor):
    p.drawRect(QRectF(8, 8, 32, 32))
    p.setBrush(c)
    p.setPen(Qt.PenStyle.NoPen)
    for x, y in ((14, 14), (26, 14), (20, 20), (14, 26), (26, 26), (32, 20), (20, 32)):
        p.drawRect(QRectF(x, y, 6, 6))


def _calib(p: QPainter, c: QColor):
    p.drawRect(QRectF(7, 7, 34, 34))
    p.setBrush(c)
    p.setPen(Qt.PenStyle.NoPen)
    s = 34 / 4
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                p.drawRect(QRectF(7 + i * s, 7 + j * s, s, s))


def _limelight(p: QPainter, c: QColor):
    p.drawRoundedRect(QRectF(7, 10, 34, 28), 6, 6)
    p.drawEllipse(QPointF(19, 24), 5.5, 5.5)
    p.setBrush(c)
    p.setPen(Qt.PenStyle.NoPen)
    for k in range(3):
        p.drawEllipse(QPointF(30 + (k % 2) * 5, 18 + k * 6), 1.8, 1.8)


def _field(p: QPainter, c: QColor):
    path = QPainterPath()
    path.moveTo(4, 34)
    path.lineTo(14, 14)
    path.lineTo(44, 14)
    path.lineTo(34, 34)
    path.closeSubpath()
    p.drawPath(path)
    p.drawLine(QPointF(24, 14), QPointF(19, 34))


def _riocam(p: QPainter, c: QColor):
    p.drawRoundedRect(QRectF(5, 9, 24, 18), 4, 4)
    p.drawEllipse(QPointF(17, 18), 4.5, 4.5)
    path = QPainterPath()  # USB cable to the roboRIO box
    path.moveTo(17, 27)
    path.cubicTo(17, 36, 30, 32, 30, 38)
    p.drawPath(path)
    p.drawRect(QRectF(25, 38, 18, 6))


def make_icons(normal: str, active: str) -> dict[str, QIcon]:
    return {
        "riocam": _icon(_riocam, normal, active),
        "camera": _icon(_camera, normal, active),
        "apriltag": _icon(_tag, normal, active),
        "calibration": _icon(_calib, normal, active),
        "limelight": _icon(_limelight, normal, active),
        "field": _icon(_field, normal, active),
    }
