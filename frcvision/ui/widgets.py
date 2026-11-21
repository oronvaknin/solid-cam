"""Reusable widgets: video view, cards, chips."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import theme


class VideoView(QWidget):
    """Paints the camera frame (aspect-fit) and lets pages draw overlays in image coordinates."""

    clicked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QImage | None = None
        self.data = None
        self.overlay = None  # callable(painter, view)
        self.placeholder = "No camera connected"
        self.show_crosshair = True
        self.setMinimumSize(360, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ------------------------------------------------------------------ API
    def set_image(self, image: QImage | None, data=None):
        self._image = image
        self.data = data
        self.update()

    def clear(self, text: str | None = None):
        self._image = None
        self.data = None
        if text:
            self.placeholder = text
        self.update()

    @property
    def image_size(self):
        return (self._image.width(), self._image.height()) if self._image is not None else (0, 0)

    def image_rect(self) -> QRectF:
        if self._image is None:
            return QRectF(self.rect())
        iw, ih = self._image.width(), self._image.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = iw * s, ih * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    @property
    def scale(self) -> float:
        if self._image is None:
            return 1.0
        return self.image_rect().width() / self._image.width()

    def map(self, x: float, y: float) -> QPointF:
        r = self.image_rect()
        s = self.scale
        return QPointF(r.x() + (x + 0.5) * s, r.y() + (y + 0.5) * s)

    # --------------------------------------------------------------- painting
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0b0c0e"))
        if self._image is None:
            p.setPen(QColor(theme.MUTED))
            f = QFont(self.font())
            f.setPointSize(12)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder)
            return
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        r = self.image_rect()
        p.drawImage(r, self._image)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self.show_crosshair:
            pen = QPen(QColor(155, 226, 45, 150), 1)
            p.setPen(pen)
            c = r.center()
            p.drawLine(QPointF(c.x() - 14, c.y()), QPointF(c.x() + 14, c.y()))
            p.drawLine(QPointF(c.x(), c.y() - 14), QPointF(c.x(), c.y() + 14))
        if self.overlay is not None and self.data is not None:
            p.save()
            try:
                self.overlay(p, self)
            finally:
                p.restore()

    def mousePressEvent(self, ev):
        if self._image is None:
            return
        r = self.image_rect()
        s = self.scale
        x = (ev.position().x() - r.x()) / s
        y = (ev.position().y() - r.y()) / s
        self.clicked.emit(x, y)


def draw_label(p: QPainter, pos: QPointF, text: str, fg=theme.ACCENT, bg="#0b0c0e", size=9, bold=True,
               align_center=True):
    f = QFont(p.font())
    f.setPointSizeF(size)
    f.setBold(bold)
    p.setFont(f)
    fm = p.fontMetrics()
    w = fm.horizontalAdvance(text) + 10
    h = fm.height() + 4
    x = pos.x() - w / 2 if align_center else pos.x()
    rect = QRectF(x, pos.y() - h, w, h)
    bgc = QColor(bg)
    bgc.setAlpha(200)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(bgc)
    p.drawRoundedRect(rect, 4, 4)
    p.setPen(QColor(fg))
    p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


class Card(QFrame):
    def __init__(self, title: str, subtitle: str | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(14, 12, 14, 14)
        self._layout.setSpacing(8)
        head = QHBoxLayout()
        t = QLabel(title)
        t.setObjectName("CardTitle")
        head.addWidget(t)
        head.addStretch(1)
        self.header_right = head
        self._layout.addLayout(head)
        if subtitle:
            s = QLabel(subtitle)
            s.setObjectName("Hint")
            s.setWordWrap(True)
            self._layout.addWidget(s)
        self._form: QFormLayout | None = None

    def form(self) -> QFormLayout:
        if self._form is None:
            self._form = QFormLayout()
            self._form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self._form.setHorizontalSpacing(12)
            self._form.setVerticalSpacing(7)
            self._layout.addLayout(self._form)
        return self._form

    def add_row(self, label: str, widget):
        lab = QLabel(label)
        lab.setObjectName("Muted")
        self.form().addRow(lab, widget)
        return widget

    def add(self, widget):
        self._form = None  # following rows start a new form below this widget
        self._layout.addWidget(widget)
        return widget

    def add_layout(self, layout):
        self._form = None
        self._layout.addLayout(layout)
        return layout


class Chip(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setObjectName("Chip")
        self.setTextFormat(Qt.TextFormat.RichText)

    def set_state(self, text: str, color: str = theme.MUTED):
        self.setText(f'<span style="color:{color}">&#9679;</span>&nbsp;{text}')


def value_label(text: str = "-") -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("Value")
    lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lab


def hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("Hint")
    lab.setWordWrap(True)
    return lab


def compact_combos(root: QWidget, min_chars: int = 12):
    """Stop long combo-box items from forcing the side panels wider than the window allows."""
    for cb in root.findChildren(QComboBox):
        cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        cb.setMinimumContentsLength(min_chars)
        cb.view().setTextElideMode(Qt.TextElideMode.ElideNone)
        cb.view().setMinimumWidth(cb.view().sizeHintForColumn(0) + 24)


def scroll_panel(width: int = 380) -> tuple[QFrame, QVBoxLayout]:
    """Right-hand settings panel with a scroll area; returns (frame, content layout)."""
    frame = QFrame()
    frame.setObjectName("Panel")
    frame.setMinimumWidth(width)
    frame.setMaximumWidth(width + 120)
    outer = QVBoxLayout(frame)
    outer.setContentsMargins(0, 0, 0, 0)
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    inner = QWidget()
    lay = QVBoxLayout(inner)
    lay.setContentsMargins(12, 12, 12, 12)
    lay.setSpacing(12)
    sa.setWidget(inner)
    outer.addWidget(sa)
    return frame, lay
