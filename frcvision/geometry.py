"""Small rigid-transform helpers.

Conventions
-----------
* Field / robot / tag frames follow WPILib: +X forward, +Y left, +Z up,
  rotations are (roll about X, pitch about Y, yaw about Z), yaw CCW-positive.
* OpenCV camera frame: +X right, +Y down, +Z forward (optical axis).
* A "WPILib camera frame" is the camera with WPILib axes: +X forward, +Y left, +Z up.
* ``T_a_b`` is a 4x4 matrix that maps points expressed in frame *b* into frame *a*.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# p_cv = CV_FROM_WPI @ p_wpi   (WPILib camera axes -> OpenCV camera axes)
CV_FROM_WPI = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ]
)

# For a tag, IPPE_SQUARE uses x = right, y = up, z = out of the tag toward the viewer.
# WPILib tag frame uses x = out of the tag (toward the viewer), y = viewer's right, z = up.
# p_tag_wpi = TAG_WPI_FROM_IPPE @ p_ippe
TAG_WPI_FROM_IPPE = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)


def make_T(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def rvec_tvec_to_T(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    return make_T(R, tvec)


def T_to_rvec_tvec(T: np.ndarray):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return rvec.reshape(3, 1), T[:3, 3].reshape(3, 1).copy()


def quat_to_R(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def euler_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  (WPILib Rotation3d convention), radians."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_euler(R: np.ndarray):
    """Inverse of :func:`euler_to_R`. Returns (roll, pitch, yaw) in radians."""
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(R[2, 0]) < 0.999999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw


def pose_to_list_deg(T: np.ndarray):
    """[x, y, z, roll, pitch, yaw] with angles in degrees (Limelight 'botpose' style)."""
    roll, pitch, yaw = R_to_euler(T[:3, :3])
    x, y, z = T[:3, 3]
    return [float(x), float(y), float(z), math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0
