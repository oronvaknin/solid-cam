"""Generate the in-match adaptive exposure robot code (Java / WPILib) from the app's tune results."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "robot_templates" / "LimelightAdaptiveExposure.java.tmpl"
CLASS_FILE = "LimelightAdaptiveExposure.java"


@dataclass
class RobotCamera:
    name: str  # NetworkTables name of the Limelight
    host: str  # IP address
    exposure_key: str = "exposure"
    baseline_raw: float = 150
    integer_values: bool = True
    mount: dict = field(default_factory=lambda: {"forward": 0.0, "left": 0.0, "up": 0.5,
                                                  "roll": 0.0, "pitch": 0.0, "yaw": 0.0})
    hfov: float = 63.3
    vfov: float = 49.7
    note: str = ""


@dataclass
class AdaptiveSettings:
    min_factor: float = 0.6
    max_factor: float = 2.5
    step: float = 1.2
    window: float = 0.75
    max_tag_distance: float = 4.5
    adapt_auto: bool = True
    adapt_teleop: bool = True


def java_ident(name: str) -> str:
    ident = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper() or "LIMELIGHT"
    return ident if not ident[0].isdigit() else "LL_" + ident


def _num(v: float) -> str:
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s if "." in s else s + ".0"


def camera_block(cam: RobotCamera, s: AdaptiveSettings, ident: str) -> str:
    m = cam.mount
    baseline = str(int(round(cam.baseline_raw))) if cam.integer_values else _num(cam.baseline_raw)
    comment = f"    /** {cam.host} - baseline {cam.baseline_raw * 0.01:.2f} ms{(' - ' + cam.note) if cam.note else ''}. */"
    return (
        f"{comment}\n"
        f"    public static final Config {ident} = new Config(\n"
        f"        \"{cam.name}\", \"{cam.host}\", \"{cam.exposure_key}\", {baseline}, {str(cam.integer_values).lower()},\n"
        f"        // robot -> camera: forward, left, up (m); roll, pitch, yaw (deg; WPILib pitch + = tilted DOWN)\n"
        f"        new Transform3d(new Translation3d({_num(m['forward'])}, {_num(m['left'])}, {_num(m['up'])}),\n"
        f"            new Rotation3d(Math.toRadians({_num(m['roll'])}), Math.toRadians({_num(-m['pitch'])}), "
        f"Math.toRadians({_num(m['yaw'])}))))\n"
        f"        .withFov({_num(cam.hfov)}, {_num(cam.vfov)})\n"
        f"        .withRange({_num(s.min_factor)}, {_num(s.max_factor)})\n"
        f"        .withStep({_num(s.step)}, {_num(s.window)})\n"
        f"        .withMaxTagDistance({_num(s.max_tag_distance)})\n"
        f"        .withModes({str(s.adapt_auto).lower()}, {str(s.adapt_teleop).lower()});\n"
    )


def unique_idents(cameras: list[RobotCamera]) -> list[str]:
    out, seen = [], set()
    for cam in cameras:
        base = java_ident(cam.name)
        ident, n = base, 2
        while ident in seen:
            ident, n = f"{base}_{n}", n + 1
        seen.add(ident)
        out.append(ident)
    return out


def generate(cameras: list[RobotCamera], settings: AdaptiveSettings, package: str = "frc.robot.vision") -> str:
    if not cameras:
        raise ValueError("No cameras to generate.")
    idents = unique_idents(cameras)
    blocks = "\n".join(camera_block(c, settings, i) for c, i in zip(cameras, idents))
    src = TEMPLATE.read_text(encoding="utf-8")
    return (src.replace("{{PACKAGE}}", package)
            .replace("{{GENERATED_AT}}", datetime.now().strftime("%Y-%m-%d %H:%M"))
            .replace("{{FIRST_CAMERA}}", idents[0])
            .replace("{{CAMERAS}}", blocks.rstrip("\n")))


def package_for(folder: Path) -> str:
    """frc.robot.vision for .../src/main/java/frc/robot/vision, etc."""
    parts = Path(folder).resolve().parts
    for i in range(len(parts) - 1):
        if parts[i].lower() == "java" and i >= 2 and parts[i - 1].lower() == "main" and parts[i - 2].lower() == "src":
            pkg = ".".join(parts[i + 1:])
            return pkg or "frc.robot"
    return "frc.robot"


def usage_snippet(cameras: list[RobotCamera]) -> str:
    idents = unique_idents(cameras)
    lines = ["// ---- fields (Robot.java or RobotContainer.java) ----"]
    for ident in idents:
        var = ident.lower().replace("_", "")
        lines.append(f"private final LimelightAdaptiveExposure {var}Exposure =\n"
                     f"    new LimelightAdaptiveExposure(LimelightAdaptiveExposure.Cameras.{ident});")
    lines.append("\n// ---- robotPeriodic() ----")
    for ident in idents:
        var = ident.lower().replace("_", "")
        lines.append(f"{var}Exposure.update(drivetrain.getPose(), drivetrain.getRobotRelativeSpeeds());")
    lines.append("\n// ---- autonomousInit() ----")
    for ident in idents:
        lines.append(f"{ident.lower().replace('_', '')}Exposure.resetToBaseline();")
    lines.append("\n// ---- optional: freeze while aiming / shooting ----")
    lines.append(f"// {idents[0].lower().replace('_', '')}Exposure.setHold(isAiming);")
    return "\n".join(lines)
