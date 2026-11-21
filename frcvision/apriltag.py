"""AprilTag 36h11 detection + per-tag and multi-tag (field) pose estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibration import CameraCalibration
from .field_layout import FieldLayout
from .geometry import (
    CV_FROM_WPI,
    TAG_WPI_FROM_IPPE,
    R_to_euler,
    inv_T,
    make_T,
    rvec_tvec_to_T,
    wrap_deg,
)

REFINE_METHODS = {
    "Sub-pixel (fast)": "subpix",
    "AprilTag (accurate, slower)": "apriltag",
    "None": "none",
}


@dataclass
class AprilTagSettings:
    tag_size_m: float = 0.1651
    downscale: float = 1.0
    refine: str = "subpix"
    id_filter: frozenset[int] | None = None
    multitag: bool = True
    max_single_tag_ambiguity: float = 0.2
    draw_cube: bool = True


@dataclass
class TagDetection:
    id: int
    corners: np.ndarray  # (4,2) TL, TR, BR, BL as printed on the tag
    center: np.ndarray
    tx: float  # deg, + = right
    ty: float  # deg, + = up
    ta: float  # % of image area
    in_layout: bool = False
    distance: float | None = None  # m
    ambiguity: float | None = None
    reproj_err: float | None = None
    T_cv_tag: np.ndarray | None = None  # IPPE tag frame -> OpenCV camera frame
    cam_xyz: tuple | None = None  # (forward, left, up) in metres, WPILib camera axes
    tag_yaw: float | None = None  # deg, 0 = tag faces the camera squarely
    cube: np.ndarray | None = None  # (8,2) projected cube points for drawing


@dataclass
class PoseEstimate:
    T_field_camera: np.ndarray  # WPILib camera axes
    T_field_robot: np.ndarray
    tag_ids: list[int]
    reproj_err: float
    method: str
    avg_dist: float
    ambiguity: float | None = None


@dataclass
class AprilTagResult:
    tags: list[TagDetection] = field(default_factory=list)
    pose: PoseEstimate | None = None
    image_size: tuple = (0, 0)
    calibrated: bool = False


class AprilTagPipeline:
    def __init__(self, settings: AprilTagSettings | None = None):
        self.settings = settings or AprilTagSettings()
        self._detector = None
        self._detector_key = None

    # ---------------------------------------------------------------- detector
    def _get_detector(self):
        key = self.settings.refine
        if self._detector is None or key != self._detector_key:
            d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            p = cv2.aruco.DetectorParameters()
            p.cornerRefinementMethod = (
                cv2.aruco.CORNER_REFINE_APRILTAG if key == "apriltag" else cv2.aruco.CORNER_REFINE_NONE
            )
            p.minMarkerPerimeterRate = 0.02
            p.polygonalApproxAccuracyRate = 0.05
            self._detector = cv2.aruco.ArucoDetector(d, p)
            self._detector_key = key
        return self._detector

    def detect_raw(self, gray: np.ndarray):
        s = self.settings
        scale = max(1.0, float(s.downscale))
        img = gray
        if scale > 1.0:
            img = cv2.resize(gray, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_AREA)
        corners, ids, _ = self._get_detector().detectMarkers(img)
        out = []
        if ids is None:
            return out
        for c, i in zip(corners, ids.ravel()):
            c = c.reshape(4, 2).astype(np.float32)
            if scale > 1.0:
                c = (c + 0.5) * scale - 0.5
            if s.refine == "subpix":
                c = cv2.cornerSubPix(
                    gray, c.reshape(-1, 1, 2), (5, 5), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                ).reshape(4, 2)
            # OpenCV's APRILTAG_36h11 bit pattern is the official tag rotated 180 deg,
            # so its "first corner" is the printed tag's bottom-right. Re-order to
            # TL, TR, BR, BL of the tag as printed / mounted on the field.
            c = np.roll(c, 2, axis=0)
            out.append((int(i), c))
        return out

    # ---------------------------------------------------------------- process
    def process(self, frame_bgr: np.ndarray, calib: CameraCalibration,
                layout: FieldLayout | None, T_robot_camera: np.ndarray) -> AprilTagResult:
        s = self.settings
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        K, dist = calib.K, calib.dist
        res = AprilTagResult(image_size=(w, h), calibrated=not calib.estimated)

        half = s.tag_size_m / 2.0
        obj_ippe = np.array(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float64
        )
        cube_local = np.array(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0],
             [-half, half, s.tag_size_m], [half, half, s.tag_size_m],
             [half, -half, s.tag_size_m], [-half, -half, s.tag_size_m]], dtype=np.float64
        )

        for tag_id, c in self.detect_raw(gray):
            if s.id_filter and tag_id not in s.id_filter:
                continue
            center = c.mean(axis=0)
            nc = cv2.undistortPoints(center.reshape(1, 1, 2).astype(np.float64), K, dist).reshape(2)
            det = TagDetection(
                id=tag_id,
                corners=c,
                center=center,
                tx=math.degrees(math.atan(nc[0])),
                ty=-math.degrees(math.atan(nc[1])),
                ta=100.0 * abs(cv2.contourArea(c)) / float(w * h),
                in_layout=bool(layout and tag_id in layout.tags),
            )
            # ---- per-tag pose (IPPE square gives both ambiguous solutions)
            try:
                n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                    obj_ippe, c.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
            except cv2.error:
                n = 0
            if n:
                errs = np.asarray(errs).ravel()
                T = rvec_tvec_to_T(rvecs[0], tvecs[0])
                det.T_cv_tag = T
                det.distance = float(np.linalg.norm(T[:3, 3]))
                det.reproj_err = float(errs[0])
                det.ambiguity = float(errs[0] / errs[1]) if n > 1 and errs[1] > 1e-9 else 0.0
                p_wpi = CV_FROM_WPI.T @ T[:3, 3]
                det.cam_xyz = (float(p_wpi[0]), float(p_wpi[1]), float(p_wpi[2]))
                R_cam_tag = CV_FROM_WPI.T @ T[:3, :3] @ TAG_WPI_FROM_IPPE.T
                det.tag_yaw = wrap_deg(math.degrees(math.atan2(R_cam_tag[1, 0], R_cam_tag[0, 0])) - 180.0)
                if s.draw_cube:
                    pts, _ = cv2.projectPoints(cube_local, rvecs[0], tvecs[0], K, dist)
                    det.cube = pts.reshape(-1, 2)
            res.tags.append(det)

        if layout is not None:
            res.pose = self._estimate_field_pose(res.tags, K, dist, layout, T_robot_camera)
        return res

    # ------------------------------------------------------------- field pose
    def _estimate_field_pose(self, tags, K, dist, layout, T_robot_camera):
        s = self.settings
        usable = [t for t in tags if t.in_layout and t.T_cv_tag is not None]
        if not usable:
            return None
        T_cv_wpicam = make_T(CV_FROM_WPI, [0, 0, 0])

        if len(usable) >= 2 and s.multitag:
            obj = np.concatenate([layout.corners_field(t.id, s.tag_size_m) for t in usable]).astype(np.float64)
            img = np.concatenate([t.corners for t in usable]).astype(np.float64)
            candidates = []
            # (a) global solver on all corners
            try:
                ok, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
                if ok:
                    candidates.append((rv, tv))
            except cv2.error:
                pass
            # (b) seeded from each tag's best single-tag solution
            for t in usable:
                T_cv_field = inv_T(self._single_tag_field_cv(t, layout))
                rv, _ = cv2.Rodrigues(T_cv_field[:3, :3])
                candidates.append((rv, T_cv_field[:3, 3].reshape(3, 1).copy()))
            best_err, best_T = float("inf"), None
            for rv, tv in candidates:
                try:
                    rv, tv = cv2.solvePnPRefineLM(obj, img, K, dist, rv.astype(np.float64), tv.astype(np.float64))
                except cv2.error:
                    continue
                proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
                e = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
                if e < best_err:
                    best_err, best_T = e, inv_T(rvec_tvec_to_T(rv, tv))
            if best_T is None:
                return None
            T_field_cv, err = best_T, best_err
            method, amb = "multi-tag", None
        else:
            best = min(usable, key=lambda t: t.ambiguity if t.ambiguity is not None else 1.0)
            if best.ambiguity is not None and best.ambiguity > s.max_single_tag_ambiguity:
                return None
            T_field_cv = self._single_tag_field_cv(best, layout)
            err, method, amb = best.reproj_err or 0.0, "single-tag", best.ambiguity
            usable = [best]

        T_field_cam = T_field_cv @ T_cv_wpicam
        T_field_robot = T_field_cam @ inv_T(T_robot_camera)
        avg_dist = float(np.mean([t.distance for t in usable]))
        return PoseEstimate(T_field_cam, T_field_robot, [t.id for t in usable], err, method, avg_dist, amb)

    @staticmethod
    def _single_tag_field_cv(tag: TagDetection, layout: FieldLayout) -> np.ndarray:
        T_field_tag = layout.tags[tag.id]
        T_tag_ippe = make_T(TAG_WPI_FROM_IPPE, [0, 0, 0])
        return T_field_tag @ T_tag_ippe @ inv_T(tag.T_cv_tag)


def robot_to_camera_T(forward: float, left: float, up: float,
                      roll_deg: float, pitch_up_deg: float, yaw_deg: float) -> np.ndarray:
    """Camera mounting pose in robot space. Pitch is given as 'tilted up' positive."""
    from .geometry import euler_to_R

    R = euler_to_R(math.radians(roll_deg), -math.radians(pitch_up_deg), math.radians(yaw_deg))
    return make_T(R, [forward, left, up])


def pose_summary(T: np.ndarray):
    roll, pitch, yaw = R_to_euler(T[:3, :3])
    x, y, z = T[:3, 3]
    return float(x), float(y), float(z), math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
