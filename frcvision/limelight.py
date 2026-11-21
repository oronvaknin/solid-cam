"""Limelight integration over its official REST API (port 5807).

* Live results / status polling (botpose_wpiblue, MegaTag2, fiducials, temperature ...).
* Read / back up / upload / restore the Limelight camera calibration (``/cal-*``).
  Limelight does not publish the calibration JSON schema, so we never invent one:
  the Limelight's own calibration JSON is used as a template, the intrinsics and
  distortion inside it are located and replaced, and every change is shown before upload.
* Field map (.fmap) export / upload.

Once uploaded, the calibration lives on the Limelight (file system) and is used by every
AprilTag pipeline automatically - no laptop / app needed during matches.
"""

from __future__ import annotations

import copy
import json
import math
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

from .calibration import CameraCalibration
from .field_layout import FieldLayout

REST_PORT = 5807
STREAM_PORT = 5800
UI_PORT = 5801
BACKUP_DIR = Path(__file__).resolve().parent.parent / "limelight_backups"

HOST_PRESETS = [
    ("limelight.local", "mDNS name (default)"),
    ("172.29.0.1", "USB-C connection"),
    ("172.28.0.1", "USB-C connection (alt)"),
]
# resolutions Limelight cameras calibrate / run at (used only if the JSON has no resolution field)
KNOWN_RESOLUTIONS = [(1280, 960), (1280, 800), (1280, 720), (960, 720), (640, 480), (640, 400), (320, 240)]


class LimelightError(Exception):
    pass


def team_to_host(team: int) -> str:
    return f"10.{team // 100}.{team % 100}.11"


# ============================================================================ client
class LimelightClient:
    def __init__(self, host: str, timeout: float = 1.5):
        self.host = host.strip().replace("http://", "").replace("https://", "").split("/")[0].split(":")[0]
        self.timeout = timeout
        self._ip: str | None = None
        # never send robot-network traffic through a system proxy
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def resolve(self, timeout: float = 2.5) -> str:
        """Resolve once (with a timeout) - mDNS names like limelight.local can take seconds on Windows."""
        if self._ip is None:
            box = {}

            def work():
                try:
                    box["ip"] = socket.gethostbyname(self.host)
                except OSError:
                    pass

            t = threading.Thread(target=work, daemon=True)
            t.start()
            t.join(timeout)
            if "ip" not in box:
                raise LimelightError(f"Cannot resolve '{self.host}'. Use the IP instead: 10.TE.AM.11 on the robot "
                                     "network, or 172.29.0.1 / 172.28.0.1 over USB.")
            self._ip = box["ip"]
        return self._ip

    @property
    def stream_url(self) -> str:
        return f"http://{self._ip or self.host}:{STREAM_PORT}"

    @property
    def ui_url(self) -> str:
        return f"http://{self._ip or self.host}:{UI_PORT}"

    def _url(self, path: str, params: dict | None = None) -> str:
        q = f"?{urllib.parse.urlencode(params)}" if params else ""
        return f"http://{self.resolve()}:{REST_PORT}{path}{q}"

    def request(self, method: str, path: str, body: bytes | None = None, params: dict | None = None,
                content_type: str = "application/json", timeout: float | None = None) -> bytes:
        req = urllib.request.Request(self._url(path, params), data=body, method=method,
                                     headers={"Content-Type": content_type} if body is not None else {})
        try:
            with self._opener.open(req, timeout=timeout or self.timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            detail = e.read()[:200].decode("utf-8", "replace") if e.fp else ""
            raise LimelightError(f"HTTP {e.code} on {method} {path} {detail}".strip()) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            reason = getattr(e, "reason", e)
            raise LimelightError(f"Cannot reach Limelight at {self.host}: {reason}") from None

    def get_json(self, path: str, **params):
        data = self.request("GET", path, params=params or None)
        if not data.strip():
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            raise LimelightError(f"{path} did not return JSON") from None

    def post_json(self, path: str, obj, timeout: float = 6.0, **params) -> bytes:
        return self.request("POST", path, json.dumps(obj).encode("utf-8"), params=params or None, timeout=timeout)

    # ---- endpoints (docs.limelightvision.io -> REST/HTTP API)
    def status(self):
        return self.get_json("/status")

    def results(self):
        return self.get_json("/results")

    def hwreport(self):
        return self.get_json("/hwreport")

    def calibration(self, kind: str):
        """kind: default | file | eeprom | latest"""
        return self.get_json(f"/cal-{kind}")

    def cal_stats(self):
        return self.get_json("/cal-stats")

    def upload_cal_file(self, data: dict):
        return self.post_json("/cal-file", data)

    def delete_cal_file(self):
        return self.request("DELETE", "/cal-file")

    def upload_fieldmap(self, fmap: dict):
        return self.post_json("/upload-fieldmap", fmap)

    def switch_pipeline(self, index: int):
        return self.request("POST", "/pipeline-switch", params={"index": index}, body=b"")


# =========================================================================== results
@dataclass
class LLFiducial:
    id: int
    tx: float
    ty: float
    ta: float  # % of image
    distance: float | None = None


@dataclass
class LLResults:
    valid: bool = False
    pipeline: int = -1
    ptype: str = ""
    tl: float = 0.0
    cl: float = 0.0
    botpose_blue: list | None = None
    botpose_mt2_blue: list | None = None
    tagcount: int = 0
    avgdist: float = 0.0
    span: float = 0.0
    stdev_mt1: list | None = None
    stdev_mt2: list | None = None
    cam_robot: list | None = None
    fiducials: list = field(default_factory=list)
    hw: dict = field(default_factory=dict)
    received: float = field(default_factory=time.monotonic)


def _pose6(v):
    if isinstance(v, list) and len(v) >= 6 and all(isinstance(x, (int, float)) for x in v[:6]):
        if any(abs(x) > 1e-9 for x in v[:6]):
            return [float(x) for x in v[:6]]
    return None


def parse_results(d: dict) -> LLResults:
    if isinstance(d, dict) and "Results" in d and isinstance(d["Results"], dict):
        d = d["Results"]  # older firmware wraps everything in "Results"
    r = LLResults()
    if not isinstance(d, dict):
        return r
    r.valid = bool(d.get("v", 0))
    try:
        r.pipeline = int(d.get("pID", -1))
    except (TypeError, ValueError):
        r.pipeline = -1
    r.ptype = str(d.get("pTYPE", ""))
    r.tl = float(d.get("tl", 0) or 0)
    r.cl = float(d.get("cl", 0) or 0)
    r.tagcount = int(d.get("botpose_tagcount", 0) or 0)
    r.avgdist = float(d.get("botpose_avgdist", 0) or 0)
    r.span = float(d.get("botpose_span", 0) or 0)
    if r.tagcount > 0:
        r.botpose_blue = _pose6(d.get("botpose_wpiblue"))
        r.botpose_mt2_blue = _pose6(d.get("botpose_orb_wpiblue"))
    r.stdev_mt1 = d.get("stdev_mt1")
    r.stdev_mt2 = d.get("stdev_mt2")
    r.cam_robot = d.get("t6c_rs")
    r.hw = d.get("hw") if isinstance(d.get("hw"), dict) else {}
    for f in d.get("Fiducial", []) or []:
        try:
            t6 = f.get("t6t_cs")
            dist = math.sqrt(sum(float(x) ** 2 for x in t6[:3])) if isinstance(t6, list) and len(t6) >= 3 else None
            if dist == 0:
                dist = None
            r.fiducials.append(LLFiducial(int(f.get("fID", -1)), float(f.get("tx", 0)), float(f.get("ty", 0)),
                                          float(f.get("ta", 0)) * (100.0 if float(f.get("ta", 0)) <= 1 else 1.0),
                                          dist))
        except (TypeError, ValueError):
            continue
    return r


def pose6_to_T(p6) -> np.ndarray:
    from .geometry import euler_to_R, make_T

    x, y, z, roll, pitch, yaw = p6
    return make_T(euler_to_R(math.radians(roll), math.radians(pitch), math.radians(yaw)), [x, y, z])


_ZOMBIES: set = set()


class LimelightPoller(QThread):
    results_ready = Signal(object)
    status_ready = Signal(object)
    connection = Signal(bool, str)

    def __init__(self, host: str, rate_hz: float = 12.0, parent=None):
        super().__init__(parent)
        self.client = LimelightClient(host, timeout=1.0)
        self.period = 1.0 / rate_hz
        self._running = False

    def stop(self):
        self._running = False
        if not self.wait(4000):  # still inside a request: never delete a running QThread
            self.setParent(None)
            _ZOMBIES.add(self)
            self.finished.connect(lambda t=self: (_ZOMBIES.discard(t), t.deleteLater()))

    def run(self):
        self._running = True
        online = None
        last_status = 0.0
        while self._running:
            t0 = time.monotonic()
            try:
                res = self.client.results()
                self.results_ready.emit(parse_results(res or {}))
                if t0 - last_status > 1.0:
                    last_status = t0
                    try:
                        self.status_ready.emit(self.client.status() or {})
                    except LimelightError:
                        pass
                if online is not True:
                    online = True
                    self.connection.emit(True, "")
            except LimelightError as exc:
                if online is not False:
                    online = False
                    self.connection.emit(False, str(exc))
                for _ in range(10):  # back off ~1 s but stay responsive to stop()
                    if not self._running:
                        break
                    time.sleep(0.1)
                continue
            dt = time.monotonic() - t0
            if dt < self.period:
                time.sleep(self.period - dt)


# ======================================================================= calibration
def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _nodes(obj, path=()):
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _nodes(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _nodes(v, path + (i,))


def _get(obj, path):
    for k in path:
        obj = obj[k]
    return obj


def _set(obj, path, value):
    for k in path[:-1]:
        obj = obj[k]
    obj[path[-1]] = value


def path_str(path) -> str:
    return "/".join(str(p) for p in path) or "(root)"


def _as_matrix(v):
    if isinstance(v, list) and len(v) == 9 and all(_is_num(x) for x in v):
        m, kind = np.array(v, float).reshape(3, 3), "flat"
    elif (isinstance(v, list) and len(v) == 3
          and all(isinstance(r, list) and len(r) == 3 and all(_is_num(x) for x in r) for r in v)):
        m, kind = np.array(v, float), "nested"
    else:
        return None
    ok = (abs(m[2, 2] - 1) < 1e-3 and abs(m[2, 0]) < 1e-6 and abs(m[2, 1]) < 1e-6 and abs(m[1, 0]) < 1e-6
          and m[0, 0] > 50 and m[1, 1] > 50 and m[0, 2] > 10 and m[1, 2] > 10)
    return (m, kind) if ok else None


@dataclass
class LLCalibInfo:
    K: np.ndarray
    dist: list
    resolution: tuple
    matrix_path: tuple | None = None
    matrix_kind: str = "flat"
    scalar_paths: dict = field(default_factory=dict)  # fx/fy/cx/cy -> path
    dist_path: tuple | None = None
    res_paths: tuple | None = None
    res_inferred: bool = False
    error_path: tuple | None = None
    error: float | None = None

    def as_calibration(self, name="limelight") -> CameraCalibration:
        w, h = self.resolution
        return CameraCalibration(w, h, self.K, np.array(self.dist[:8] if self.dist else np.zeros(5)),
                                 rms=self.error if self.error is not None else float("nan"), camera_name=name)


_RES_KEYS = [("resx", "resy"), ("res_x", "res_y"), ("width", "height"), ("image_width", "image_height"),
             ("imagewidth", "imageheight"), ("w", "h"), ("cols", "rows"), ("xres", "yres")]


def analyze_limelight_calibration(d) -> LLCalibInfo:
    if not isinstance(d, (dict, list)):
        raise LimelightError("Calibration JSON is empty.")
    mats = [(p, _as_matrix(v)) for p, v in _nodes(d)]
    mats = [(p, m) for p, m in mats if m is not None]
    info = None
    if mats:
        path, (K, kind) = mats[0]
        info = LLCalibInfo(K=K, dist=[], resolution=(0, 0), matrix_path=path, matrix_kind=kind)
    else:  # separate fx / fy / cx / cy numbers
        for p, v in _nodes(d):
            if isinstance(v, dict):
                low = {str(k).lower(): k for k in v}
                if all(k in low for k in ("fx", "fy", "cx", "cy")) and all(_is_num(v[low[k]]) for k in ("fx", "fy", "cx", "cy")):
                    fx, fy, cx, cy = (float(v[low[k]]) for k in ("fx", "fy", "cx", "cy"))
                    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
                    info = LLCalibInfo(K=K, dist=[], resolution=(0, 0))
                    info.scalar_paths = {k: p + (low[k],) for k in ("fx", "fy", "cx", "cy")}
                    break
    if info is None:
        raise LimelightError("Could not find a camera matrix in the Limelight calibration JSON.")

    # scalar fx/fy/cx/cy duplicates (kept in sync)
    if not info.scalar_paths:
        for p, v in _nodes(d):
            if isinstance(v, dict):
                low = {str(k).lower(): k for k in v}
                for k in ("fx", "fy", "cx", "cy"):
                    if k in low and _is_num(v[low[k]]):
                        info.scalar_paths[k] = p + (low[k],)

    # distortion
    cands = []
    for p, v in _nodes(d):
        if isinstance(v, list) and len(v) in (4, 5, 8, 12, 14) and all(_is_num(x) for x in v) and p:
            key = str(p[-1]).lower()
            score = 2 if re.search(r"dist|coef", key) else 0
            if info.matrix_path and p[:-1] == info.matrix_path[:-1]:
                score += 1
            if all(abs(float(x)) < 100 for x in v) and score > 0:
                cands.append((score, p, v))
    if cands:
        cands.sort(key=lambda c: -c[0])
        _, info.dist_path, dv = cands[0]
        info.dist = [float(x) for x in dv]

    # resolution
    cx, cy = info.K[0, 2], info.K[1, 2]

    def plausible(w, h):
        return 100 <= w <= 5000 and 80 <= h <= 4000 and abs(cx - w / 2) < 0.25 * w and abs(cy - h / 2) < 0.25 * h

    for p, v in _nodes(d):
        if not isinstance(v, dict):
            continue
        low = {str(k).lower(): k for k in v}
        for kx, ky in _RES_KEYS:
            if kx in low and ky in low and _is_num(v[low[kx]]) and _is_num(v[low[ky]]):
                w, h = int(v[low[kx]]), int(v[low[ky]])
                if plausible(w, h):
                    info.resolution = (w, h)
                    info.res_paths = (p + (low[kx],), p + (low[ky],))
                    break
        if info.res_paths:
            break
        for k in ("resolution", "res", "imagesize", "image_size", "size"):
            if k in low:
                rv = v[low[k]]
                if isinstance(rv, list) and len(rv) == 2 and all(_is_num(x) for x in rv) and plausible(*rv):
                    info.resolution = (int(rv[0]), int(rv[1]))
                    info.res_paths = (p + (low[k], 0), p + (low[k], 1))
                    break
        if info.res_paths:
            break
    if not info.res_paths:
        best = min(KNOWN_RESOLUTIONS, key=lambda r: abs(cx - r[0] / 2) / r[0] + abs(cy - r[1] / 2) / r[1])
        if plausible(*best) and abs(cx - best[0] / 2) < 0.1 * best[0] and abs(cy - best[1] / 2) < 0.1 * best[1]:
            info.resolution = best
        else:
            info.resolution = (int(round(2 * cx)), int(round(2 * cy)))
        info.res_inferred = True

    for p, v in _nodes(d):
        if p and _is_num(v) and re.search(r"reproj|rms", str(p[-1]).lower()):
            info.error_path, info.error = p, float(v)
            break
    return info


def build_limelight_calibration(template: dict, calib: CameraCalibration):
    """Return (new_json, changes[(path, old, new)], warnings[str], target_resolution)."""
    info = analyze_limelight_calibration(template)
    w, h = info.resolution
    warnings = []
    if abs(w / h - calib.width / calib.height) > 0.01:
        raise LimelightError(
            f"Aspect ratio mismatch: our calibration is {calib.width}x{calib.height} "
            f"({calib.width / calib.height:.3f}) but the Limelight calibration is {w}x{h} ({w / h:.3f}).\n"
            "Calibrate from the Limelight's own stream (same aspect ratio) and try again.")
    if (w, h) != (calib.width, calib.height):
        warnings.append(f"Intrinsics scaled from {calib.width}x{calib.height} to the Limelight's {w}x{h}.")
    if info.res_inferred:
        warnings.append(f"The Limelight JSON has no explicit resolution - assumed {w}x{h} from its principal point.")
    c = calib.for_resolution(w, h)
    new = copy.deepcopy(template)
    changes = []

    if info.matrix_path is not None:
        K = c.K.tolist()
        val = [x for row in K for x in row] if info.matrix_kind == "flat" else K
        changes.append((path_str(info.matrix_path), _get(template, info.matrix_path), val))
        _set(new, info.matrix_path, val)
    for k, p in info.scalar_paths.items():
        v = {"fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy}[k]
        changes.append((path_str(p), _get(template, p), v))
        _set(new, p, v)

    if info.dist_path is None:
        raise LimelightError("Could not find distortion coefficients in the Limelight calibration JSON.")
    n = len(info.dist)
    ours = [float(x) for x in np.asarray(c.dist).ravel()]
    if n < 5:
        raise LimelightError(f"Limelight stores only {n} distortion coefficients; ours has {len(ours)}.")
    if len(ours) > n:
        if any(abs(x) > 1e-9 for x in ours[n:]):
            warnings.append(f"Limelight keeps {n} distortion coefficients - extra ones dropped "
                            "(use the Standard 5-coefficient model).")
        ours = ours[:n]
    ours = ours + [0.0] * (n - len(ours))
    changes.append((path_str(info.dist_path), info.dist, ours))
    _set(new, info.dist_path, ours)

    if info.error_path is not None and c.rms == c.rms:
        changes.append((path_str(info.error_path), info.error, c.rms))
        _set(new, info.error_path, c.rms)
    return new, changes, warnings, (w, h)


def is_custom_calibration(d) -> bool:
    try:
        analyze_limelight_calibration(d)
        return True
    except LimelightError:
        return False


def save_backup(host: str, kind: str, data) -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9.]+", "-", host)
    path = BACKUP_DIR / f"{safe}_{kind}_{stamp}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# ========================================================================= field map
def layout_to_fmap(layout: FieldLayout) -> dict:
    """Limelight .fmap: field-centre origin (same axes as wpiblue), row-major 4x4, size in mm."""
    fids = []
    for tid in sorted(layout.tags):
        T = layout.tags[tid].copy()
        T[0, 3] -= layout.length / 2.0
        T[1, 3] -= layout.width / 2.0
        fids.append({
            "family": "apriltag3_36h11_classic",
            "id": int(tid),
            "size": round(layout.tag_size * 1000.0, 2),
            "transform": [float(x) for x in T.reshape(-1)],
            "unique": 1,
        })
    return {"fiducials": fids, "type": "frc"}
