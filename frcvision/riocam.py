"""Control a USB camera plugged into the roboRIO - without touching robot code.

WPILib's CameraServer (``CameraServer.startAutomaticCapture()``) runs a cscore MJPEG server on the
roboRIO (port 1181 for the first camera, 1182 for the next ...). Besides the video stream, that
server accepts camera-setting commands over plain HTTP (verified in wpilibsuite/allwpilib v2026.2.1,
cscore/src/main/native/cpp/MjpegServerImpl.cpp):

    GET /settings.json                         -> {"controls": [...], "modes": [...]}
    GET /config.json                           -> current source config (mode, brightness, ...)
    GET /?action=command&brightness=40&...     -> sets camera properties directly
    GET /stream.mjpg?resolution=320x240&fps=15&compression=50   -> stream (per-connection scaling)

The robot program does not see or handle any of this - cscore applies it to the camera. Settings
stay on the camera until it loses power (or robot code sets them again).
Ports 1180-1190 are open on the FRC field network, so this also works from the driver station.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

PORTS = list(range(1181, 1191))
PROFILES_PATH = Path(__file__).resolve().parent.parent / "rio_camera_profiles.json"
KINDS = {1: "bool", 2: "int", 4: "string", 8: "enum"}


class RioCamError(Exception):
    pass


def rio_hosts(team: int) -> list[str]:
    out = []
    if team > 0:
        out += [f"10.{team // 100}.{team % 100}.2", f"roboRIO-{team}-FRC.local"]
    out.append("172.22.11.2")  # roboRIO over its USB-B port
    return out


@dataclass
class CamControl:
    name: str
    kind: str  # bool | int | enum | string
    min: int = 0
    max: int = 0
    step: int = 1
    default: str = ""
    value: str = ""
    menu: dict = field(default_factory=dict)  # int -> label

    @property
    def raw(self) -> bool:
        return self.name.startswith("raw_")

    def int_value(self) -> int:
        try:
            return int(float(self.value))
        except ValueError:
            return 0


@dataclass
class RioCamera:
    host: str
    port: int
    controls: list = field(default_factory=list)
    modes: list = field(default_factory=list)
    config: dict | None = None

    @property
    def label(self) -> str:
        m = self.config or {}
        mode = f"{m.get('width', '?')}x{m.get('height', '?')} {m.get('fps', '?')}fps" if m else ""
        return f"port {self.port}  {mode}".strip()

    def stream_url(self, resolution: str | None = None, fps: int | None = None, compression: int | None = None):
        q = {}
        if resolution:
            q["resolution"] = resolution
        if fps:
            q["fps"] = fps
        if compression is not None and compression >= 0:
            q["compression"] = compression
        qs = ("?" + "&".join(f"{k}={v}" for k, v in q.items())) if q else ""
        return f"http://{self.host}:{self.port}/stream.mjpg{qs}"

    def control(self, name: str) -> CamControl | None:
        return next((c for c in self.controls if c.name == name), None)


class CsCoreClient:
    def __init__(self, host: str, port: int, timeout: float = 1.5):
        self.host = host.strip().replace("http://", "").split("/")[0].split(":")[0]
        self.port = port
        self.timeout = timeout
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _get(self, path: str) -> str:
        url = f"http://{self.host}:{self.port}{path}"
        try:
            with self._opener.open(url, timeout=self.timeout) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            raise RioCamError(f"HTTP {e.code} from {url}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise RioCamError(f"Cannot reach {self.host}:{self.port} ({getattr(e, 'reason', e)})") from None

    def settings(self):
        return parse_settings(self._get("/settings.json"))

    def config(self) -> dict | None:
        try:
            return json.loads(self._get("/config.json"), strict=False)
        except (RioCamError, json.JSONDecodeError):
            return None

    def set(self, values: dict) -> str:
        """Set camera properties. Returns cscore's per-property response text."""
        if not values:
            return ""
        parts = [f"{_esc(str(k))}={_esc(str(v))}" for k, v in values.items()]
        resp = self._get("/?action=command&" + "&".join(parts))
        ignored = [ln.split(":")[0] for ln in resp.splitlines() if "ignored" in ln or "invalid" in ln]
        if ignored:
            raise RioCamError(f"Camera ignored: {', '.join(ignored)}")
        return resp

    def fetch(self) -> RioCamera:
        controls, modes = self.settings()
        return RioCamera(self.host, self.port, controls, modes, self.config())


def _esc(s: str) -> str:
    # cscore only accepts [A-Za-z0-9_-=&%./] in the query and URL-decodes it
    return urllib.parse.quote(s, safe="").replace("~", "%7E")


def parse_settings(text: str):
    try:
        data = json.loads(text, strict=False)
    except json.JSONDecodeError:
        # cscore writes this JSON by hand; a string property with a quote in it can break it
        text = re.sub(r'("value": ")(.*?)("\s*[,}])', lambda m: m.group(1) + m.group(2).replace('"', "'") + m.group(3),
                      text)
        data = json.loads(text, strict=False)

    def i(v, d=0):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return d

    controls = []
    for c in data.get("controls", []):
        kind = KINDS.get(i(c.get("type")), None)
        if kind is None:
            continue
        menu = {}
        if isinstance(c.get("menu"), dict):
            menu = {i(k): str(v).strip() for k, v in c["menu"].items() if str(v).strip()}
        controls.append(CamControl(str(c.get("name", "")), kind, i(c.get("min")), i(c.get("max")),
                                   max(1, i(c.get("step"), 1)), str(c.get("default", "")),
                                   str(c.get("value", "")), menu))
    modes = []
    for m in data.get("modes", []):
        modes.append({"pixelFormat": m.get("pixelFormat", ""), "width": i(m.get("width")),
                      "height": i(m.get("height")), "fps": i(m.get("fps"))})
    return controls, modes


def scan(host: str, ports=PORTS, timeout: float = 1.2) -> list[RioCamera]:
    """Find every CameraServer MJPEG server on the host (ports are probed in parallel)."""
    found: dict[int, RioCamera] = {}
    lock = threading.Lock()

    def probe(port):
        try:
            cam = CsCoreClient(host, port, timeout).fetch()
        except (RioCamError, json.JSONDecodeError, ValueError):
            return
        with lock:
            found[port] = cam

    threads = [threading.Thread(target=probe, args=(p,), daemon=True) for p in ports]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout + 1.5)
    return [found[p] for p in sorted(found)]


# ===================================================================== presets & profiles
def _pct(cam: RioCamera, name: str, pct: float):
    c = cam.control(name)
    if c is None:
        return None
    v = c.min + (c.max - c.min) * pct / 100.0
    return int(round(v / c.step) * c.step) if c.step else int(v)


def preset_values(cam: RioCamera, preset: str) -> dict:
    """Common USB-webcam presets, built only from controls this camera really has."""
    has = {c.name for c in cam.controls}
    v: dict = {}
    if preset == "auto":
        if "exposure_auto" in has:
            v["exposure_auto"] = 3  # aperture priority = auto
        if "white_balance_temperature_auto" in has:
            v["white_balance_temperature_auto"] = 1
        for n in ("brightness", "contrast", "saturation", "gain", "sharpness"):
            c = cam.control(n)
            if c is not None and c.default != "":
                v[n] = int(float(c.default))
    elif preset == "fast":  # short exposure for a moving robot, brightness/gain make up for it
        if "exposure_auto" in has:
            v["exposure_auto"] = 1  # manual
        for n, p in (("exposure_absolute", 8), ("gain", 70), ("brightness", 60), ("sharpness", 60)):
            val = _pct(cam, n, p)
            if val is not None:
                v[n] = val
    elif preset == "dim":  # dark venue: longer exposure, moderate gain
        if "exposure_auto" in has:
            v["exposure_auto"] = 1
        for n, p in (("exposure_absolute", 35), ("gain", 45), ("brightness", 55)):
            val = _pct(cam, n, p)
            if val is not None:
                v[n] = val
    if "focus_auto" in has and preset in ("fast", "dim"):
        v["focus_auto"] = 0  # autofocus hunting makes the image pump while driving
    return v


def current_values(cam: RioCamera) -> dict:
    return {c.name: (c.value if c.kind == "string" else c.int_value())
            for c in cam.controls if not c.raw}


def load_profiles() -> dict:
    try:
        return json.loads(PROFILES_PATH.read_text(encoding="utf-8")).get("profiles", {})
    except (OSError, json.JSONDecodeError):
        return {}


def save_profile(name: str, values: dict):
    profiles = load_profiles()
    profiles[name] = values
    PROFILES_PATH.write_text(json.dumps({"profiles": profiles}, indent=2), encoding="utf-8")


def delete_profile(name: str):
    profiles = load_profiles()
    profiles.pop(name, None)
    PROFILES_PATH.write_text(json.dumps({"profiles": profiles}, indent=2), encoding="utf-8")


def apply_values(client: CsCoreClient, cam: RioCamera, values: dict) -> list[str]:
    """Apply in a safe order (auto modes first, so manual values are not overridden)."""
    names = {c.name for c in cam.controls}
    usable = {k: v for k, v in values.items() if k in names}
    skipped = [k for k in values if k not in names]
    first = {k: usable.pop(k) for k in list(usable) if k.endswith("_auto") or k == "exposure_auto"}
    if first:
        client.set(first)
    if usable:
        client.set(usable)
    return skipped
