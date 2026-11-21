"""FRC AprilTag field layouts.

Supports
* WPILib ``AprilTagFieldLayout`` JSON (``{"tags": [...], "field": {"length", "width"}}``)
  in the blue-alliance-origin convention used by WPILib 2026 / Limelight ``botpose_wpiblue``.
* Limelight ``.fmap`` files (field-center origin, 4x4 transforms) - converted to blue origin.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import make_T, quat_to_R

DEFAULT_TAG_SIZE_M = 0.1651  # 6.5 in, FRC 2024+ (36h11)
DEFAULT_FIELD = (16.541, 8.069)  # 2026 REBUILT

FIELDS_DIR = Path(__file__).resolve().parent.parent / "fields"


@dataclass
class FieldLayout:
    name: str
    length: float
    width: float
    tags: dict[int, np.ndarray] = field(default_factory=dict)  # id -> T_field_tag (4x4)
    tag_size: float = DEFAULT_TAG_SIZE_M
    path: str = ""

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path) -> "FieldLayout":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if "fiducials" in data:
            return cls._from_fmap(data, path)
        return cls._from_wpilib(data, path)

    @classmethod
    def _from_wpilib(cls, data: dict, path: Path) -> "FieldLayout":
        f = data.get("field", {})
        layout = cls(
            name=data.get("name", path.stem),
            length=float(f.get("length", DEFAULT_FIELD[0])),
            width=float(f.get("width", DEFAULT_FIELD[1])),
            tag_size=float(data.get("tagSizeMeters", DEFAULT_TAG_SIZE_M)),
            path=str(path),
        )
        for tag in data["tags"]:
            tr = tag["pose"]["translation"]
            q = tag["pose"]["rotation"]["quaternion"]
            R = quat_to_R(q["W"], q["X"], q["Y"], q["Z"])
            layout.tags[int(tag["ID"])] = make_T(R, [tr["x"], tr["y"], tr["z"]])
        return layout

    @classmethod
    def _from_fmap(cls, data: dict, path: Path) -> "FieldLayout":
        length, width = DEFAULT_FIELD
        layout = cls(name=path.stem, length=length, width=width, path=str(path))
        sizes = []
        for fid in data["fiducials"]:
            T = np.array(fid["transform"], dtype=float).reshape(4, 4)
            T[0, 3] += length / 2.0  # fmap origin is the field center
            T[1, 3] += width / 2.0
            layout.tags[int(fid["id"])] = T
            if "size" in fid:
                sizes.append(float(fid["size"]) / 1000.0)
        if sizes:
            layout.tag_size = float(np.median(sizes))
        return layout

    # ------------------------------------------------------------------ geometry
    def corners_field(self, tag_id: int, size: float | None = None) -> np.ndarray:
        """Field-frame 3D corners of a tag, ordered as the viewer sees them:
        top-left, top-right, bottom-right, bottom-left."""
        s = (size or self.tag_size) / 2.0
        # WPILib tag frame: +X out of the face, +Y = viewer's right, +Z up
        local = np.array(
            [
                [0.0, -s, s, 1.0],  # TL
                [0.0, s, s, 1.0],  # TR
                [0.0, s, -s, 1.0],  # BR
                [0.0, -s, -s, 1.0],  # BL
            ]
        )
        return (self.tags[tag_id] @ local.T).T[:, :3]

    def tag_xy_yaw(self, tag_id: int):
        T = self.tags[tag_id]
        yaw = math.atan2(T[1, 0], T[0, 0])
        return float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), yaw


def bundled_layouts() -> list[Path]:
    if not FIELDS_DIR.exists():
        return []
    files = sorted(FIELDS_DIR.glob("*.json"), reverse=True)  # newest season first
    return files + sorted(FIELDS_DIR.glob("*.fmap"))
