"""Automatic lighting (exposure) tuning for Limelights over the official REST API.

Principle
---------
For AprilTags the best exposure is the *shortest* one at which every tag in view is still
detected in (almost) every frame with low corner noise: shorter exposure = less motion blur
while the robot drives. A safety margin is then added for darker spots on the field.

Measurements come from the Limelight itself (/results): which tags it detects in each frame
and how much tx/ty jitter (corner noise) there is while the robot stands still.

Safety
------
During the sweep settings are applied *without* `flush`, so nothing is written to the
Limelight's disk until the very end. Stopping, an error, or even a reboot leaves the original
pipeline untouched. Only the final value is saved (flush=1), and it can be reverted.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .limelight import LimelightClient, LimelightError

EXPOSURE_UNIT_MS = 0.01  # docs.limelightvision.io: "exposure setting in .01 millisecond intervals"
PROFILES_PATH = Path(__file__).resolve().parent.parent / "lighting_profiles.json"


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


@dataclass
class TuneSettings:
    min_ms: float = 0.1
    max_ms: float = 10.0
    steps: int = 12
    dwell_s: float = 1.0
    settle_s: float = 0.4
    sample_hz: float = 25.0
    min_reliability: float = 0.95
    margin: float = 1.5
    refine: bool = True
    save: bool = True


@dataclass
class LightKeys:
    exposure: str
    gain: str | None
    black: str | None
    original: dict

    def describe(self) -> str:
        parts = [f"{k} = {v}" for k, v in self.original.items()]
        return ", ".join(parts)


def find_light_keys(pipe) -> LightKeys:
    """Locate the exposure / gain / black-level keys in a Limelight pipeline JSON."""
    if isinstance(pipe, dict) and isinstance(pipe.get("pipeline"), dict):
        pipe = pipe["pipeline"]
    if not isinstance(pipe, dict) or not pipe:
        raise LimelightError("The Limelight returned no pipeline settings.")

    def pick(pattern, exclude=()):
        for k, v in pipe.items():
            kl = str(k).lower()
            if re.search(pattern, kl) and not any(e in kl for e in exclude) and _is_num(v):
                return k
        return None

    exp = pick(r"exposure", exclude=("auto", "comp", "target"))
    if exp is None:
        keys = ", ".join(list(map(str, pipe.keys()))[:25])
        raise LimelightError(f"No exposure setting found in the pipeline JSON (keys: {keys} ...).")
    gain = pick(r"gain")
    black = pick(r"black")
    return LightKeys(exp, gain, black, {k: pipe[k] for k in (exp, gain, black) if k is not None})


@dataclass
class SweepPoint:
    ms: float
    raw: float
    frames: int = 0
    counts: list = field(default_factory=list)
    reliability: float = 0.0
    median_tags: float = 0.0
    jitter_deg: float | None = None
    jitter_m: float | None = None
    ids: list = field(default_factory=list)
    passed: bool = False
    phase: str = "sweep"  # sweep | refine | verify


@dataclass
class TuneResult:
    host: str
    pipeline: int = -1
    keys: LightKeys | None = None
    points: list = field(default_factory=list)
    expected_tags: int = 0
    jitter_limit: float | None = None
    min_pass_ms: float | None = None
    recommended_ms: float | None = None
    recommended_raw: float | None = None
    verify: SweepPoint | None = None
    saved: bool = False
    ok: bool = False
    message: str = ""


class Cancelled(Exception):
    pass


class AutoLightTuner:
    def __init__(self, host: str, settings: TuneSettings, on_event=None, should_stop=lambda: False):
        self.client = LimelightClient(host, timeout=1.5)
        self.s = settings
        self.on_event = on_event or (lambda kind, **kw: None)
        self.should_stop = should_stop
        self.result = TuneResult(host=self.client.host)
        self._int_values = True

    # ------------------------------------------------------------- helpers
    def _check(self):
        if self.should_stop():
            raise Cancelled()

    def _sleep(self, secs):
        end = time.monotonic() + secs
        while time.monotonic() < end:
            self._check()
            time.sleep(0.02)

    def _raw(self, ms: float):
        raw = ms / EXPOSURE_UNIT_MS
        return int(round(raw)) if self._int_values else round(raw, 2)

    def _apply(self, values: dict, flush=False):
        params = {"flush": 1} if flush else {}
        self.client.post_json("/update-pipeline", values, **params)

    def _sample(self, point: SweepPoint):
        period = 1.0 / self.s.sample_hz
        end = time.monotonic() + self.s.dwell_s
        last_frame = None
        txs: dict[int, list] = {}
        poses = []
        while time.monotonic() < end:
            self._check()
            t0 = time.monotonic()
            d = self.client.results() or {}
            if isinstance(d.get("Results"), dict):
                d = d["Results"]
            frame = d.get("fidx", d.get("ts"))
            if frame is None or frame != last_frame:
                last_frame = frame
                fids = d.get("Fiducial") or []
                ids = []
                for f in fids:
                    try:
                        i = int(f.get("fID"))
                        txs.setdefault(i, []).append((float(f.get("tx", 0)), float(f.get("ty", 0))))
                        ids.append(i)
                    except (TypeError, ValueError):
                        continue
                point.counts.append(len(set(ids)))
                if int(d.get("botpose_tagcount", 0) or 0) > 0:
                    bp = d.get("botpose_wpiblue")
                    if isinstance(bp, list) and len(bp) >= 2:
                        poses.append((float(bp[0]), float(bp[1])))
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
        point.frames = len(point.counts)
        point.median_tags = float(np.median(point.counts)) if point.counts else 0.0
        steady = [v for v in txs.values() if len(v) >= max(3, 0.5 * point.frames)]
        point.ids = sorted(i for i, v in txs.items() if len(v) >= max(3, 0.5 * point.frames))
        if steady:
            point.jitter_deg = float(np.mean([math.hypot(np.std([a for a, _ in v]), np.std([b for _, b in v]))
                                              for v in steady]))
        if len(poses) >= 3:
            arr = np.array(poses)
            point.jitter_m = float(math.hypot(arr[:, 0].std(), arr[:, 1].std()))

    def _measure(self, ms: float, phase="sweep") -> SweepPoint:
        raw = self._raw(ms)
        p = SweepPoint(ms=raw * EXPOSURE_UNIT_MS, raw=raw, phase=phase)
        self.on_event("status", text=f"{self.result.host}: exposure {p.ms:.2f} ms ({phase})")
        self._apply({self.result.keys.exposure: raw})
        self._sleep(self.s.settle_s)
        self._sample(p)
        return p

    def _evaluate(self, points):
        """(Re)compute expected tag count, jitter floor and pass/fail for every point."""
        expected = int(max((p.median_tags for p in points), default=0))
        for p in points:
            p.reliability = (sum(1 for c in p.counts if c >= expected) / len(p.counts)) if p.counts and expected else 0.0
        reliable = [p.jitter_deg for p in points if p.reliability >= self.s.min_reliability and p.jitter_deg is not None]
        floor = min(reliable) if reliable else None
        limit = max(2.0 * floor, floor + 0.02) if floor is not None else None
        for p in points:
            p.passed = (expected > 0 and p.reliability >= self.s.min_reliability
                        and (limit is None or (p.jitter_deg is not None and p.jitter_deg <= limit)))
        return expected, limit

    # ----------------------------------------------------------------- run
    def run(self) -> TuneResult:
        r = self.result
        s = self.s
        try:
            self.on_event("status", text=f"{r.host}: reading pipeline…")
            res = self.client.results() or {}
            if isinstance(res.get("Results"), dict):
                res = res["Results"]
            r.pipeline = int(res.get("pID", 0) or 0)
            pipe = self.client.get_json("/pipeline-atindex", index=r.pipeline)
            r.keys = find_light_keys(pipe)
            self._int_values = isinstance(r.keys.original[r.keys.exposure], int)
            self.on_event("keys", host=r.host, pipeline=r.pipeline, keys=r.keys)

            # 1. coarse sweep, short -> long exposure
            n = max(3, s.steps)
            grid = [s.min_ms * (s.max_ms / s.min_ms) ** (i / (n - 1)) for i in range(n)]
            pts: list[SweepPoint] = []
            for ms in grid:
                p = self._measure(ms)
                pts.append(p)
                self._evaluate(pts)
                self.on_event("point", host=r.host, point=p, points=list(pts))
                # past the top of the usable range (image saturating): nothing more to learn
                if (len(pts) >= 3 and any(q.passed for q in pts[:-2])
                        and pts[-1].reliability < 0.5 and pts[-2].reliability < 0.5):
                    break
            r.expected_tags, r.jitter_limit = self._evaluate(pts)
            if r.expected_tags == 0:
                raise LimelightError("No AprilTags were detected at any exposure - point the Limelight at the "
                                     "tags (typical match distance) and keep the robot still.")
            first = next((i for i, p in enumerate(pts) if p.passed), None)
            if first is None:
                raise LimelightError("No exposure gave reliable, low-noise detection. Get closer to the tags or "
                                     "widen the exposure range.")

            # 2. refine the threshold between the last failing and the first passing step
            lo = pts[first - 1].ms if first > 0 else None
            hi = pts[first].ms
            if s.refine and lo is not None:
                for _ in range(2):
                    mid = math.sqrt(lo * hi)
                    if self._raw(mid) in (self._raw(lo), self._raw(hi)):
                        break
                    p = self._measure(mid, "refine")
                    pts.append(p)
                    r.expected_tags, r.jitter_limit = self._evaluate(pts)
                    self.on_event("point", host=r.host, point=p, points=list(pts))
                    if p.passed:
                        hi = p.ms
                    else:
                        lo = p.ms
            r.points = sorted(pts, key=lambda q: q.ms)
            r.min_pass_ms = hi

            # 3. recommendation = threshold x margin, but stay inside the contiguous passing range
            upper = hi
            for q in r.points:
                if q.ms < hi:
                    continue
                if not q.passed:
                    break
                upper = q.ms
            rec = min(hi * s.margin, upper)
            r.recommended_raw = self._raw(rec)
            r.recommended_ms = r.recommended_raw * EXPOSURE_UNIT_MS

            # 4. verify at the recommended value, then save
            v = self._measure(r.recommended_ms, "verify")
            v.reliability = (sum(1 for c in v.counts if c >= r.expected_tags) / len(v.counts)) if v.counts else 0.0
            v.passed = v.reliability >= s.min_reliability
            r.verify = v
            self.on_event("point", host=r.host, point=v, points=r.points + [v])
            if s.save:
                self._apply({r.keys.exposure: r.recommended_raw}, flush=True)
                r.saved = True
            r.ok = v.passed
            r.message = (f"exposure {r.recommended_ms:.2f} ms (threshold {r.min_pass_ms:.2f} ms x{s.margin:g}), "
                         f"{r.expected_tags} tags, reliability {v.reliability * 100:.0f}%"
                         + ("" if v.passed else " - verification below target, check the result"))
            if pts[0].passed:
                r.message += " | even the shortest exposure tested works - you can lower 'min'."
        except Cancelled:
            self._restore()
            r.message = "Stopped - original exposure restored."
        except LimelightError as exc:
            self._restore()
            r.message = str(exc)
        self.on_event("done", host=r.host, result=r)
        return r

    def _restore(self):
        if self.result.keys is not None:
            try:
                self._apply({self.result.keys.exposure: self.result.keys.original[self.result.keys.exposure]})
            except LimelightError:
                pass


def revert(result: TuneResult):
    """Write the original light settings back and save them."""
    if result.keys is None:
        raise LimelightError("Nothing to revert.")
    LimelightClient(result.host).post_json("/update-pipeline", result.keys.original, flush=1)


# ======================================================================== profiles
def load_profiles() -> dict:
    try:
        return json.loads(PROFILES_PATH.read_text(encoding="utf-8")).get("profiles", {})
    except (OSError, json.JSONDecodeError):
        return {}


def save_profile(name: str, results: list[TuneResult]):
    profiles = load_profiles()
    entry = {"created": datetime.now().isoformat(timespec="seconds"), "limelights": {}}
    for r in results:
        if r.keys is None or r.recommended_raw is None:
            continue
        values = {r.keys.exposure: r.recommended_raw}
        for k in (r.keys.gain, r.keys.black):
            if k:
                values[k] = r.keys.original[k]
        entry["limelights"][r.host] = {"pipeline": r.pipeline, "values": values,
                                       "exposure_ms": r.recommended_ms}
    profiles[name] = entry
    PROFILES_PATH.write_text(json.dumps({"profiles": profiles}, indent=2), encoding="utf-8")
    return entry


def delete_profile(name: str):
    profiles = load_profiles()
    profiles.pop(name, None)
    PROFILES_PATH.write_text(json.dumps({"profiles": profiles}, indent=2), encoding="utf-8")


def apply_profile(name: str) -> list[str]:
    """Apply a saved profile to every Limelight in it. Returns one message per Limelight."""
    entry = load_profiles().get(name)
    if not entry:
        raise LimelightError(f"Profile '{name}' not found.")
    out = []
    for host, cfg in entry["limelights"].items():
        c = LimelightClient(host, timeout=1.5)
        try:
            res = c.results() or {}
            if isinstance(res.get("Results"), dict):
                res = res["Results"]
            pid = int(res.get("pID", -1))
            if pid != cfg["pipeline"]:
                out.append(f"{host}: skipped - active pipeline is {pid}, profile was made on pipeline {cfg['pipeline']}")
                continue
            c.post_json("/update-pipeline", cfg["values"], flush=1)
            out.append(f"{host}: exposure {cfg.get('exposure_ms', 0):.2f} ms applied and saved")
        except LimelightError as exc:
            out.append(f"{host}: {exc}")
    return out
