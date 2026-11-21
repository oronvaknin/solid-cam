"""Shared application state (read by the vision thread, written by the UI)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Signal

from .apriltag import AprilTagSettings
from .calibration import CameraCalibration
from .field_layout import FieldLayout


class AppState(QObject):
    calibration_changed = Signal()
    layout_changed = Signal()
    camera_opened = Signal(dict)

    def __init__(self):
        super().__init__()
        self.camera_name = ""
        self.frame_size = (0, 0)
        self.calibration: CameraCalibration | None = None
        self.calibration_path: str = ""
        self.use_calibration = True
        self.hfov_guess = 70.0
        self.layout: FieldLayout | None = None
        self.apriltag = AprilTagSettings()
        self.T_robot_camera = np.eye(4)
        self._cache_key = None
        self._cache_val = None

    def set_calibration(self, calib: CameraCalibration | None, path: str = ""):
        self.calibration = calib
        self.calibration_path = path
        self._cache_key = None
        self.calibration_changed.emit()

    def set_layout(self, layout: FieldLayout | None):
        self.layout = layout
        if layout is not None:
            self.apriltag.tag_size_m = layout.tag_size
        self.layout_changed.emit()

    @property
    def has_real_calibration(self) -> bool:
        return self.calibration is not None and self.use_calibration

    def intrinsics(self, width: int, height: int) -> CameraCalibration:
        key = (width, height, id(self.calibration), self.use_calibration, self.hfov_guess)
        if key != self._cache_key:
            if self.has_real_calibration:
                val = self.calibration.for_resolution(width, height)
            else:
                val = CameraCalibration.estimate(width, height, self.hfov_guess)
            self._cache_key, self._cache_val = key, val
        return self._cache_val
