"""Camera capture thread.

The worker grabs frames, runs the active processor (raw / AprilTag / ChArUco) in the
same background thread and hands the newest packet to the UI ("latest frame wins",
so a slow UI never builds up latency).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

BACKENDS = {
    "DirectShow": cv2.CAP_DSHOW,
    "Media Foundation": cv2.CAP_MSMF,
    "Auto": cv2.CAP_ANY,
}

RESOLUTIONS = [(640, 480), (800, 600), (1280, 720), (1280, 800), (1280, 960), (1600, 1200), (1920, 1080)]
FPS_OPTIONS = [30, 60, 90, 120]


@dataclass
class CameraConfig:
    index: int = 0
    name: str = "Camera 0"
    backend: str = "DirectShow"
    width: int = 1280
    height: int = 720
    fps: int = 30
    mjpg: bool = True
    url: str = ""  # if set: network MJPEG/RTSP stream instead of a USB camera


@dataclass
class FramePacket:
    image: QImage
    frame: np.ndarray  # raw BGR frame
    result: object
    mode: str
    fps: float
    latency_ms: float
    timestamp: float = field(default_factory=time.perf_counter)


_RETIRED: set = set()


def retire(worker: "VisionWorker", wait_ms: int = 1500):
    """Stop a worker without freezing the UI and without ever deleting a running QThread
    (that aborts the whole app). A worker still blocked in a read is kept alive until it ends."""
    try:
        worker.packet_ready.disconnect()
    except (RuntimeError, TypeError):
        pass
    if worker.stop(wait_ms):
        worker.deleteLater()
        return
    worker.setParent(None)  # must not be destroyed together with its page / window
    _RETIRED.add(worker)
    worker.finished.connect(lambda w=worker: (_RETIRED.discard(w), w.deleteLater()))


def wait_retired(timeout_ms: int = 4000):
    """At app exit: give retired workers a chance to finish (reads time out after 3 s)."""
    deadline = time.monotonic() + timeout_ms / 1000
    for w in list(_RETIRED):
        w.wait(max(0, int((deadline - time.monotonic()) * 1000)))


def list_camera_names() -> list[str]:
    try:
        from PySide6.QtMultimedia import QMediaDevices

        return [d.description() for d in QMediaDevices.videoInputs()]
    except Exception:  # QtMultimedia missing / no backend
        return []


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    if frame.ndim == 2:
        h, w = frame.shape
        return QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_Grayscale8).copy()
    h, w = frame.shape[:2]
    return QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_BGR888).copy()


class VisionWorker(QThread):
    packet_ready = Signal()
    opened = Signal(dict)
    failed = Signal(str)

    def __init__(self, config: CameraConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.mode = "raw"
        self.processors: dict = {}  # mode -> callable(frame) -> (display_frame, result)
        # Never reset once set: stop() may arrive while the camera / stream is still opening.
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: FramePacket | None = None
        self._pending = False
        self._prop_queue: list[tuple] = []
        self.frame_count = 0

    # --------------------------------------------------------------- UI side
    def take_latest(self) -> FramePacket | None:
        with self._lock:
            pkt, self._latest = self._latest, None
            self._pending = False
        return pkt

    def set_property(self, prop: int, value: float):
        with self._lock:
            self._prop_queue.append(("set", prop, value))

    def open_driver_settings(self):
        with self._lock:
            self._prop_queue.append(("set", cv2.CAP_PROP_SETTINGS, 1))

    def request_stop(self):
        self._stop_event.set()

    def stop(self, timeout_ms: int = 5000) -> bool:
        """Ask the thread to finish and wait. Returns False if it is still running (e.g. blocked in a read)."""
        self._stop_event.set()
        return self.wait(timeout_ms)

    # ----------------------------------------------------------- worker side
    def _open(self) -> cv2.VideoCapture | None:
        c = self.config
        if c.url:
            params = []
            for name in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
                if hasattr(cv2, name):  # never block for long on an unreachable robot
                    params += [getattr(cv2, name), 3000]
            try:
                cap = cv2.VideoCapture(c.url, cv2.CAP_FFMPEG, params)
            except (cv2.error, TypeError):
                cap = cv2.VideoCapture(c.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap.release()
                return None
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap = cv2.VideoCapture(c.index, BACKENDS.get(c.backend, cv2.CAP_ANY))
        if not cap.isOpened():
            cap.release()
            return None
        if c.mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)
        cap.set(cv2.CAP_PROP_FPS, c.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self):
        cap = self._open()
        if self._stop_event.is_set():  # stopped while opening
            if cap is not None:
                cap.release()
            return
        if cap is None:
            src = self.config.url or f"camera {self.config.index} ({self.config.backend})"
            self.failed.emit(f"Could not open {src}.")
            return
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            self.failed.emit("Camera opened but returned no frames (is it used by another app?).")
            return
        h, w = frame.shape[:2]
        self.opened.emit({
            "width": w, "height": h,
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0),
            "backend": cap.getBackendName() if hasattr(cap, "getBackendName") else self.config.backend,
            "exposure": float(cap.get(cv2.CAP_PROP_EXPOSURE)),
            "gain": float(cap.get(cv2.CAP_PROP_GAIN)),
            "brightness": float(cap.get(cv2.CAP_PROP_BRIGHTNESS)),
        })

        fails = 0
        fps = 0.0
        t_prev = time.perf_counter()
        while not self._stop_event.is_set():
            with self._lock:
                props, self._prop_queue = self._prop_queue, []
            for _, prop, value in props:
                try:
                    cap.set(prop, value)
                except cv2.error:
                    pass

            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails > 50:
                    self.failed.emit("Camera stopped delivering frames (disconnected?).")
                    break
                time.sleep(0.01)
                continue
            fails = 0
            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 * (1.0 / dt)

            mode = self.mode
            proc = self.processors.get(mode)
            t0 = time.perf_counter()
            display, result = frame, None
            if proc is not None:
                try:
                    display, result = proc(frame)
                except Exception as exc:  # never kill the capture loop
                    result = exc
            latency = (time.perf_counter() - t0) * 1000.0
            pkt = FramePacket(bgr_to_qimage(display), frame, result, mode, fps, latency)
            self.frame_count += 1
            with self._lock:
                self._latest = pkt
                notify = not self._pending
                self._pending = True
            if notify:
                self.packet_ready.emit()
        cap.release()
