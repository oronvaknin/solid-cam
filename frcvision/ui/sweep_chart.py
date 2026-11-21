"""Exposure-sweep chart: two small multiples sharing a log exposure axis (no dual axis).

Top: detection reliability (%).  Bottom: tag jitter (deg).  Passing points are filled,
failing points hollow (identity never by colour alone); hover shows the exact values.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from . import theme

SURFACE = QColor(theme.PANEL_2)
GRID = QColor("#30353d")
INK = QColor(theme.TEXT)
MUTED = QColor(theme.MUTED)
SERIES = QColor(theme.ACCENT)
NICE_MS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]


class SweepChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.min_rel = 0.95
        self.jitter_limit = None
        self.recommended = None
        self.range_ms = (0.1, 10.0)
        self._hover = None
        self.setMouseTracking(True)
        self.setMinimumHeight(270)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_data(self, points, min_rel=0.95, jitter_limit=None, recommended=None, range_ms=None):
        self.points = sorted([p for p in points if p.phase != "verify"], key=lambda p: p.ms)
        self.min_rel = min_rel
        self.jitter_limit = jitter_limit
        self.recommended = recommended
        if range_ms:
            self.range_ms = range_ms
        self.update()

    def clear(self):
        self.set_data([])

    # ------------------------------------------------------------ geometry
    def _xr(self):
        lo, hi = self.range_ms
        if self.points:
            lo = min(lo, self.points[0].ms)
            hi = max(hi, self.points[-1].ms)
        return math.log10(lo), math.log10(hi)

    PAD = 9  # headroom so markers at the maximum never touch the panel edge

    def _plot_rects(self):
        left, right, top = 46, 14, 22
        h = self.height()
        gap = 34
        ph = (h - top - gap - 40) / 2
        w = self.width() - left - right
        return QRectF(left, top, w, ph), QRectF(left, top + ph + gap, w, ph)

    def _y(self, v, vmax, rect):
        return rect.bottom() - min(v, vmax) / vmax * (rect.height() - self.PAD)

    def _x(self, ms, rect):
        a, b = self._xr()
        return rect.left() + (math.log10(ms) - a) / max(b - a, 1e-9) * rect.width()

    # ------------------------------------------------------------ events
    def mouseMoveEvent(self, ev):
        if not self.points:
            return
        r1, _ = self._plot_rects()
        x = ev.position().x()
        best = min(self.points, key=lambda p: abs(self._x(p.ms, r1) - x))
        self._hover = best if abs(self._x(best.ms, r1) - x) < 26 else None
        self.update()

    def leaveEvent(self, _ev):
        self._hover = None
        self.update()

    # ------------------------------------------------------------ paint
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        f = QFont(self.font())
        f.setPointSizeF(8)
        p.setFont(f)
        r1, r2 = self._plot_rects()
        if not self.points:
            p.setPen(MUTED)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Run the auto-tune to see the exposure sweep here")
            return
        jit_vals = [q.jitter_deg for q in self.points if q.jitter_deg is not None]
        jmax = max(jit_vals + ([self.jitter_limit] if self.jitter_limit else []) + [0.05]) * 1.15
        jmax = _nice_ceiling(jmax)

        self._panel(p, r1, "Detection reliability", [0, 50, 100], lambda v: f"{v:.0f}%",
                    lambda q: q.reliability * 100, 100.0,
                    self.min_rel * 100, f"target {self.min_rel * 100:.0f}%")
        self._panel(p, r2, "Tag jitter (deg, lower is better)", [0, jmax / 2, jmax], lambda v: f"{v:.2f}",
                    lambda q: q.jitter_deg, jmax,
                    self.jitter_limit, "limit" if self.jitter_limit else None)

        # shared x ticks below the bottom panel, axis title on its own line under them
        a, b = self._xr()
        p.setPen(MUTED)
        for ms in NICE_MS:
            if a - 1e-9 <= math.log10(ms) <= b + 1e-9:
                x = self._x(ms, r2)
                p.drawText(QRectF(x - 20, r2.bottom() + 4, 40, 14), Qt.AlignmentFlag.AlignCenter, f"{ms:g}")
        p.drawText(QRectF(r2.left(), r2.bottom() + 20, r2.width(), 14), Qt.AlignmentFlag.AlignCenter,
                   "exposure (ms, log scale)")

        # recommended exposure: one hairline across both panels; label sits inside the top panel's
        # lower area, beside the line (that region is empty: short exposures fail there)
        if self.recommended:
            x = self._x(self.recommended, r1)
            p.setPen(QPen(INK, 1))
            p.drawLine(QPointF(x, r1.top() + self.PAD - 4), QPointF(x, r2.bottom()))
            label = f"recommended {self.recommended:.2f} ms"
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(label) + 10
            lx = x + 6 if x + 6 + tw <= r1.right() else x - 6 - tw
            box = QRectF(lx, r1.bottom() - 34, tw, 16)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(SURFACE)
            p.drawRoundedRect(box, 3, 3)
            p.setPen(INK)
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, label)

        if self._hover is not None:
            self._tooltip(p, self._hover, r1, r2)

    def _panel(self, p, rect, title, ticks, fmt, getter, vmax, ref, ref_label):
        p.setPen(MUTED)
        p.drawText(QRectF(rect.left(), rect.top() - 18, rect.width(), 14),
                   Qt.AlignmentFlag.AlignLeft, title)
        for t in ticks:
            y = self._y(t, vmax, rect)
            p.setPen(QPen(GRID, 1))
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            p.setPen(MUTED)
            p.drawText(QRectF(0, y - 7, rect.left() - 6, 14), Qt.AlignmentFlag.AlignRight, fmt(t))
        if ref is not None:
            y = self._y(ref, vmax, rect)
            p.setPen(QPen(MUTED, 1))
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            if ref_label:  # left end: short exposures detect nothing, so no marks live there
                fm = p.fontMetrics()
                tw = fm.horizontalAdvance(ref_label) + 8
                near_top = y - rect.top() < 18  # e.g. 95% sits right under 100%: label goes below
                box = QRectF(rect.left() + 4, y + 2 if near_top else y - 15, tw, 13)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(SURFACE)
                p.drawRect(box)
                p.setPen(MUTED)
                p.drawText(box, Qt.AlignmentFlag.AlignCenter, ref_label)

        pts = [(q, getter(q)) for q in self.points]
        xy = [(q, QPointF(self._x(q.ms, rect), self._y(v, vmax, rect))) for q, v in pts if v is not None]
        if len(xy) > 1:
            path = QPainterPath(xy[0][1])
            for _, pt in xy[1:]:
                path.lineTo(pt)
            pen = QPen(SERIES, 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        for q, pt in xy:
            # 2px surface ring, then the marker: filled = passes, hollow = fails
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(SURFACE)
            p.drawEllipse(pt, 6.5, 6.5)
            if q.passed:
                p.setBrush(SERIES)
                p.drawEllipse(pt, 4.5, 4.5)
            else:
                p.setPen(QPen(SERIES, 2))
                p.setBrush(SURFACE)
                p.drawEllipse(pt, 3.6, 3.6)

    def _tooltip(self, p, q, r1, r2):
        x = self._x(q.ms, r1)
        p.setPen(QPen(QColor(231, 234, 238, 90), 1))
        p.drawLine(QPointF(x, r1.top()), QPointF(x, r2.bottom()))
        lines = [f"{q.ms:.2f} ms  ({q.phase})",
                 f"reliability {q.reliability * 100:.0f}%  ({q.median_tags:.0f} tags)",
                 "jitter " + ("-" if q.jitter_deg is None else f"{q.jitter_deg:.3f}°")
                 + ("" if q.jitter_m is None else f",  pose {q.jitter_m * 100:.1f} cm"),
                 "✓ passes" if q.passed else "✗ fails"]
        fm = p.fontMetrics()
        w = max(fm.horizontalAdvance(s) for s in lines) + 16
        h = len(lines) * (fm.height() + 1) + 10
        bx = x + 12 if x + 12 + w < self.width() else x - 12 - w
        box = QRectF(bx, r1.top() + 4, w, h)
        p.setPen(QPen(GRID, 1))
        p.setBrush(QColor(15, 17, 20, 235))
        p.drawRoundedRect(box, 6, 6)
        p.setPen(INK)
        for i, s in enumerate(lines):
            p.drawText(QRectF(box.left() + 8, box.top() + 5 + i * (fm.height() + 1), w, fm.height()),
                       Qt.AlignmentFlag.AlignLeft, s)


def _nice_ceiling(v: float) -> float:
    exp = math.floor(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        c = m * 10 ** exp
        if c >= v:
            return c
    return 10 ** (exp + 1)
