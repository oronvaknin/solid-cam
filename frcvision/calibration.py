"""ChArUco camera calibration (the same approach Limelight / PhotonVision use)."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

CALIB_DIR = Path(__file__).resolve().parent.parent / "calibrations"

ARUCO_DICTS = {
    name: getattr(cv2.aruco, name)
    for name in [
        "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
        "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
        "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
        "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
        "DICT_ARUCO_ORIGINAL",
    ]
    if hasattr(cv2.aruco, name)
}


# ============================================================================ board
@dataclass
class BoardConfig:
    squares_x: int = 9
    squares_y: int = 6
    square_mm: float = 25.0
    marker_mm: float = 18.0
    dictionary: str = "DICT_5X5_1000"
    legacy: bool = False  # older calib.io / OpenCV<4.6 boards with an even number of rows

    @property
    def num_markers(self) -> int:
        return (self.squares_x * self.squares_y) // 2

    def detection_dictionary(self) -> str:
        """Smallest predefined dictionary of the same family that still holds every marker
        on the board. OpenCV's DICT_NxN_50/100/250 are prefixes of DICT_NxN_1000, so this
        detects the exact same board - but identification is ~6x faster."""
        m = re.fullmatch(r"DICT_(\dX\d)_(\d+)", self.dictionary)
        if not m:
            return self.dictionary
        for n in (50, 100, 250, 1000):
            cand = f"DICT_{m.group(1)}_{n}"
            if self.num_markers <= n <= int(m.group(2)) and cand in ARUCO_DICTS:
                return cand
        return self.dictionary

    def make_board(self, for_detection: bool = False) -> "cv2.aruco.CharucoBoard":
        name = self.detection_dictionary() if for_detection else self.dictionary
        d = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[name])
        board = cv2.aruco.CharucoBoard(
            (int(self.squares_x), int(self.squares_y)),
            self.square_mm / 1000.0,
            self.marker_mm / 1000.0,
            d,
        )
        board.setLegacyPattern(bool(self.legacy))
        return board

    @property
    def total_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def size_mm(self):
        return self.squares_x * self.square_mm, self.squares_y * self.square_mm

    def describe(self) -> str:
        w, h = self.size_mm
        return (
            f"ChArUco {self.squares_x}x{self.squares_y} | square {self.square_mm:g} mm | "
            f"marker {self.marker_mm:g} mm | {self.dictionary.replace('DICT_', '')} | "
            f"board {w:g} x {h:g} mm"
        )

    def render(self, px_per_mm: float, margin_mm: float = 0.0) -> np.ndarray:
        board = self.make_board()
        w_mm, h_mm = self.size_mm
        w = int(round((w_mm + 2 * margin_mm) * px_per_mm))
        h = int(round((h_mm + 2 * margin_mm) * px_per_mm))
        return board.generateImage((w, h), marginSize=int(round(margin_mm * px_per_mm)), borderBits=1)


BOARD_PRESETS: dict[str, BoardConfig] = {
    "Printable A4/Letter 9x6 - 25 mm (default)": BoardConfig(9, 6, 25.0, 18.0, "DICT_5X5_1000"),
    # docs.limelightvision.io: calib.io_charuco_260x200_8x11_20_15_DICT_5X5.pdf, dictionary 5x5_100
    "Limelight default - calib.io 11x8, 20/15 mm": BoardConfig(11, 8, 20.0, 15.0, "DICT_5X5_100"),
    "PhotonVision default 8x8 - 1 in, 4X4": BoardConfig(8, 8, 25.4, 19.05, "DICT_4X4_1000"),
    "Large poster 12x9 - 60 mm, 5X5": BoardConfig(12, 9, 60.0, 45.0, "DICT_5X5_1000"),
}


# ===================================================================== calibration
@dataclass
class CameraCalibration:
    width: int
    height: int
    K: np.ndarray
    dist: np.ndarray
    rms: float = float("nan")
    num_images: int = 0
    camera_name: str = ""
    created: str = ""
    board: dict = field(default_factory=dict)
    std_intrinsics: list = field(default_factory=list)
    estimated: bool = False  # True for the default "no calibration" guess

    # ---------------------------------------------------------------- helpers
    @property
    def fx(self): return float(self.K[0, 0])

    @property
    def fy(self): return float(self.K[1, 1])

    @property
    def cx(self): return float(self.K[0, 2])

    @property
    def cy(self): return float(self.K[1, 2])

    @property
    def pixel_aspect(self) -> float:
        """fy / fx - Limelight calls 0.99..1.01 acceptable."""
        return self.fy / self.fx if self.fx else float("nan")

    def fov_deg(self):
        h = 2 * math.degrees(math.atan2(self.width / 2.0, self.fx))
        v = 2 * math.degrees(math.atan2(self.height / 2.0, self.fy))
        d = 2 * math.degrees(math.atan(math.hypot(self.width / 2.0 / self.fx, self.height / 2.0 / self.fy)))
        return h, v, d

    def for_resolution(self, width: int, height: int) -> "CameraCalibration":
        """Scale intrinsics to another resolution (valid only for the same sensor crop / aspect)."""
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        K = self.K.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        c = CameraCalibration(**{**self.__dict__, "width": width, "height": height, "K": K})
        return c

    @staticmethod
    def estimate(width: int, height: int, hfov_deg: float = 70.0) -> "CameraCalibration":
        f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        K = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])
        return CameraCalibration(width, height, K, np.zeros(5), estimated=True,
                                 camera_name=f"estimate ({hfov_deg:g} deg HFOV)")

    # ------------------------------------------------------------------- I/O
    def to_dict(self) -> dict:
        h, v, d = self.fov_deg()
        return {
            "format": "frc-vision-lab-calibration/1",
            "camera": self.camera_name,
            "createdAt": self.created,
            "resolution": {"width": self.width, "height": self.height},
            "cameraMatrix": self.K.tolist(),
            "distCoeffs": np.asarray(self.dist).ravel().tolist(),
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "fovDeg": {"horizontal": h, "vertical": v, "diagonal": d},
            "rmsReprojectionErrorPx": self.rms,
            "numImages": self.num_images,
            "stdDevIntrinsics": self.std_intrinsics,
            "board": self.board,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraCalibration":
        return cls(
            width=int(d["resolution"]["width"]),
            height=int(d["resolution"]["height"]),
            K=np.array(d["cameraMatrix"], dtype=float),
            dist=np.array(d["distCoeffs"], dtype=float),
            rms=float(d.get("rmsReprojectionErrorPx", float("nan"))),
            num_images=int(d.get("numImages", 0)),
            camera_name=d.get("camera", ""),
            created=d.get("createdAt", ""),
            board=d.get("board", {}),
            std_intrinsics=d.get("stdDevIntrinsics", []),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CameraCalibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower() or "camera"


def default_calibration_path(camera_name: str, width: int, height: int) -> Path:
    return CALIB_DIR / f"{slug(camera_name)}_{width}x{height}.json"


# ======================================================================= detection
@dataclass
class CharucoDetection:
    charuco_corners: np.ndarray | None  # (N,1,2) float32
    charuco_ids: np.ndarray | None  # (N,1) int32
    marker_corners: tuple
    marker_ids: np.ndarray | None
    image_size: tuple  # (w, h)
    sharpness: float = 0.0
    legacy_hint: bool = False  # markers found, corners only with the other 'legacy' setting

    @property
    def markers(self) -> int:
        return 0 if self.marker_ids is None else int(len(self.marker_ids))

    @property
    def count(self) -> int:
        return 0 if self.charuco_ids is None else int(len(self.charuco_ids))

    def points(self) -> np.ndarray:
        if self.charuco_corners is None:
            return np.zeros((0, 2), np.float32)
        return self.charuco_corners.reshape(-1, 2)

    def id_map(self) -> dict[int, np.ndarray]:
        if self.charuco_ids is None:
            return {}
        return {int(i): p for i, p in zip(self.charuco_ids.ravel(), self.points())}


class CharucoPipeline:
    def __init__(self, config: BoardConfig):
        self.set_board(config)

    def set_board(self, config: BoardConfig):
        self.config = config
        self.board = config.make_board(for_detection=True)
        dp = cv2.aruco.DetectorParameters()
        dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE  # ChArUco refines corners itself
        cp = cv2.aruco.CharucoParameters()
        cp.tryRefineMarkers = True
        self.detector = cv2.aruco.CharucoDetector(self.board, cp, dp)
        alt = BoardConfig(**{**config.__dict__, "legacy": not config.legacy}).make_board(for_detection=True)
        self._alt_detector = cv2.aruco.CharucoDetector(alt, cp, dp)
        self._last_alt_try = 0.0
        self._legacy_hint = False

    def detect(self, frame_bgr: np.ndarray) -> CharucoDetection:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        cc, ci, mc, mi = self.detector.detectBoard(gray)
        h, w = gray.shape[:2]
        n_corners = 0 if ci is None else len(ci)
        n_markers = 0 if mi is None else len(mi)
        if n_markers >= 6 and n_corners < 4:
            now = time.monotonic()
            if now - self._last_alt_try > 1.0:  # cheap: at most once per second
                self._last_alt_try = now
                _, ci2, _, _ = self._alt_detector.detectBoard(gray)
                self._legacy_hint = ci2 is not None and len(ci2) >= 6
        elif n_corners >= 6:
            self._legacy_hint = False
        sharp = 0.0
        if ci is not None and len(ci) >= 4:
            x, y, bw, bh = cv2.boundingRect(cc.reshape(-1, 2).astype(np.float32))
            x0, y0 = max(0, x), max(0, y)
            roi = gray[y0:y0 + max(bh, 1), x0:x0 + max(bw, 1)]
            if roi.size > 0:
                sharp = float(cv2.Laplacian(roi, cv2.CV_64F).var())
        return CharucoDetection(cc, ci, mc, mi, (w, h), sharp, self._legacy_hint)


# ======================================================================== session
@dataclass
class Capture:
    corners: np.ndarray  # (N,1,2)
    ids: np.ndarray  # (N,1)
    thumbnail: np.ndarray  # small BGR image
    image_size: tuple
    timestamp: float = field(default_factory=time.time)
    error: float | None = None  # per-view RMS after calibration
    T_cam_board: np.ndarray | None = None  # board pose in the OpenCV camera frame (for the 3D view)

    def estimate_pose(self, board, K, dist):
        op, ip = board.matchImagePoints(self.corners, self.ids)
        if op is None or len(op) < 6:
            return
        ok, rv, tv = cv2.solvePnP(op.astype(np.float64), ip.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if ok:
            R, _ = cv2.Rodrigues(rv)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, tv.ravel()
            self.T_cam_board = T


class CalibrationSession:
    GRID = (12, 8)  # coverage grid columns x rows

    def __init__(self):
        self.captures: list[Capture] = []

    def clear(self):
        self.captures.clear()

    @property
    def image_size(self):
        return self.captures[0].image_size if self.captures else None

    def add(self, det: CharucoDetection, frame_bgr: np.ndarray) -> Capture:
        if self.captures and det.image_size != self.image_size:
            raise ValueError("Resolution changed - clear the captures first.")
        h, w = frame_bgr.shape[:2]
        tw = 192
        thumb = cv2.resize(frame_bgr, (tw, max(1, int(h * tw / w))), interpolation=cv2.INTER_AREA)
        cap = Capture(det.charuco_corners.copy(), det.charuco_ids.copy(), thumb, det.image_size)
        self.captures.append(cap)
        return cap

    def remove(self, index: int):
        if 0 <= index < len(self.captures):
            del self.captures[index]

    def all_points(self) -> np.ndarray:
        if not self.captures:
            return np.zeros((0, 2), np.float32)
        return np.concatenate([c.corners.reshape(-1, 2) for c in self.captures])

    def coverage_grid(self) -> np.ndarray:
        gx, gy = self.GRID
        grid = np.zeros((gy, gx), int)
        if not self.captures:
            return grid
        w, h = self.image_size
        pts = self.all_points()
        cx = np.clip((pts[:, 0] / w * gx).astype(int), 0, gx - 1)
        cy = np.clip((pts[:, 1] / h * gy).astype(int), 0, gy - 1)
        np.add.at(grid, (cy, cx), 1)
        return grid

    def coverage(self) -> float:
        g = self.coverage_grid()
        return float((g > 0).mean()) if g.size else 0.0

    def novelty(self, det: CharucoDetection) -> float:
        """Smallest mean corner displacement (as a fraction of image width) to any capture."""
        if not self.captures or det.count == 0:
            return 1.0
        cur = det.id_map()
        w = det.image_size[0]
        best = 1.0
        for c in self.captures:
            prev = {int(i): p for i, p in zip(c.ids.ravel(), c.corners.reshape(-1, 2))}
            common = set(cur) & set(prev)
            if len(common) < max(4, 0.5 * len(cur)):
                continue  # mostly different corners -> new view
            d = np.mean([np.linalg.norm(cur[i] - prev[i]) for i in common]) / w
            best = min(best, float(d))
        return best

    def calibrate(self, board: "cv2.aruco.CharucoBoard", rational_model: bool = False,
                  fix_aspect: bool = False, zero_tangent: bool = False) -> CameraCalibration:
        if len(self.captures) < 4:
            raise ValueError("Need at least 4 captures (15-30 recommended).")
        obj_pts, img_pts = [], []
        used = []
        for i, c in enumerate(self.captures):
            op, ip = board.matchImagePoints(c.corners, c.ids)
            if op is None or len(op) < 6:
                continue
            obj_pts.append(op.reshape(-1, 1, 3).astype(np.float32))
            img_pts.append(ip.reshape(-1, 1, 2).astype(np.float32))
            used.append(i)
        if len(obj_pts) < 4:
            raise ValueError("Not enough usable captures (each needs 6+ corners).")

        w, h = self.image_size
        flags = 0
        if rational_model:
            flags |= cv2.CALIB_RATIONAL_MODEL
        if zero_tangent:
            flags |= cv2.CALIB_ZERO_TANGENT_DIST
        K0 = None
        if fix_aspect:
            flags |= cv2.CALIB_FIX_ASPECT_RATIO
            K0 = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-7)
        rms, K, dist, rvecs, tvecs, std_in, _std_ex, per_view = cv2.calibrateCameraExtended(
            obj_pts, img_pts, (w, h), K0, None, flags=flags, criteria=criteria
        )
        for c in self.captures:
            c.error = None
        for i, err, rv, tv in zip(used, np.asarray(per_view).ravel(), rvecs, tvecs):
            cap = self.captures[i]
            cap.error = float(err)
            R, _ = cv2.Rodrigues(rv)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, np.asarray(tv).ravel()
            cap.T_cam_board = T

        n_dist = 8 if rational_model else 5
        return CameraCalibration(
            width=w, height=h, K=K, dist=np.asarray(dist).ravel()[:n_dist],
            rms=float(rms), num_images=len(obj_pts),
            created=datetime.now().isoformat(timespec="seconds"),
            std_intrinsics=np.asarray(std_in).ravel()[:4 + n_dist].tolist(),
        )
