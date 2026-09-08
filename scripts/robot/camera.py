"""UVC exposure control for the driver station camera.

WHY THIS EXISTS
    Auto-exposure hunts as the turret pans across a bright arena, and two things break.
    The classical detector gates on absolute brightness - the tuned config uses
    value_min=200 - so a moving exposure slides the image under a fixed threshold and the
    gate fires inconsistently frame to frame. And auto-exposure lengthens exposure TIME in
    dim scenes, which is what actually produces motion blur; shutter type has nothing to
    do with it. A locked, hand-tuned exposure fixes both.

WHY NOT OpenCV
    cv2.CAP_PROP_EXPOSURE is unreliable on the AVFoundation backend, and on Logitech
    hardware specifically it is a long-standing OpenCV issue - the call reports success
    and changes nothing. The control has to be driven at the UVC layer instead.

PLATFORMS
    macOS is the deployment target and is the implemented path, via `uvc-util`. Linux
    would use v4l2-ctl and Windows has no reliable path at all. Both are reported as
    unsupported rather than silently doing nothing, because an exposure lock that quietly
    failed looks exactly like one that worked - right up until the arena lighting changes.

CONTROL NAMES VARY BY UNIT
    uvc-util exposes whatever the device's UVC descriptors declare, and the spelling is
    not consistent across cameras or uvc-util versions. Nothing here hardcodes a name:
    candidates are matched against the device's own advertised list, and a resolution
    failure reports which names were tried and what the device actually offers.

VERIFY, THEN RE-APPLY
    Opening a device can reset its controls, so the lock is applied before the open, read
    back after it, and re-applied if the open cleared it. A lock that is set and never
    verified is a lock you do not have.

Selftest:
    python scripts/robot/camera.py --selftest
Inspect a real device:
    python scripts/robot/camera.py -I 0
"""

from __future__ import annotations

import shutil
import subprocess
import sys

UVC_UTIL = "uvc-util"

# Tried in order against whatever the device advertises. The first of each is the name the
# deployment note observed; the rest are spellings seen on other units and uvc-util builds.
AUTO_MODE_CANDIDATES = (
    "auto-exposure-mode",
    "auto_exposure_mode",
    "exposure-mode",
    "ae-mode",
)
EXPOSURE_ABS_CANDIDATES = (
    "exposure-time-abs",
    "exposure_time_abs",
    "exposure-time-absolute",
    "exposure-abs",
)

# UVC CT_AE_MODE_CONTROL is a bitmap, not an enum: 1 = manual, 2 = auto, 4 = shutter
# priority, 8 = aperture priority. Manual is the one that stops the hunting.
AE_MANUAL = 1


def supported() -> bool:
    """True when this platform has an implemented exposure-lock path."""
    return sys.platform == "darwin"


def tool_available() -> bool:
    return shutil.which(UVC_UTIL) is not None


def _run(args: list[str], timeout: float = 5.0) -> tuple[int, str]:
    """(returncode, combined output). Never raises on a non-zero exit."""
    try:
        done = subprocess.run(
            [UVC_UTIL, *args], capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        return 127, f"{UVC_UTIL} not found on PATH"
    except subprocess.TimeoutExpired:
        return 124, f"{UVC_UTIL} timed out after {timeout}s"
    except OSError as exc:
        return 1, str(exc)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def parse_controls(text: str) -> list[str]:
    """Control names out of `uvc-util -c` output, lowercased.

    Shared with the selftest rather than duplicated there: builds differ in whether they
    prefix a bullet or an index, and a parser only exercised against one format is a
    parser that breaks on the unit you actually bought.
    """
    names = []
    for line in text.splitlines():
        # Strip the bullet or index BEFORE splitting. Splitting first takes "*" as the
        # token on a bulleted line and then strips it to nothing, dropping the control.
        stripped = line.strip().lstrip("*-0123456789.) ").strip()
        if not stripped:
            continue
        names.append(stripped.split()[0].lower())
    return names


def list_controls(index: int) -> list[str]:
    code, out = _run(["-I", str(index), "-c"])
    return parse_controls(out) if code == 0 else []


def resolve(candidates: tuple[str, ...], available: list[str]) -> str | None:
    """First candidate the device actually offers."""
    lookup = {name.lower() for name in available}
    for name in candidates:
        if name.lower() in lookup:
            return name
    return None


def get_control(index: int, control: str) -> str | None:
    code, out = _run(["-I", str(index), "-g", control])
    return out.strip() if code == 0 else None


def set_control(index: int, control: str, value) -> bool:
    code, _ = _run(["-I", str(index), "-s", f"{control}={value}"])
    return code == 0


def apply_lock(index: int, exposure_abs: int | None = None) -> dict:
    """Put the device into manual exposure, optionally at a fixed exposure time.

    Returns a report rather than raising, so the caller decides whether a partial lock is
    fatal. `exposure_abs=None` means manual mode at whatever exposure the device currently
    holds - which is the useful state before the value has been tuned against arena
    lighting, because it still stops the hunting.
    """
    report = {
        "platform_ok": supported(), "tool_ok": tool_available(),
        "auto_control": None, "exposure_control": None,
        "auto_set": False, "exposure_set": False, "controls": [], "error": None,
    }
    if not report["platform_ok"]:
        report["error"] = (f"no exposure-lock path implemented for {sys.platform!r}; "
                           "macOS uses uvc-util, Linux would use v4l2-ctl")
        return report
    if not report["tool_ok"]:
        report["error"] = (f"{UVC_UTIL} not on PATH - install it, or the camera hunts its "
                           "own exposure while the turret pans")
        return report

    available = list_controls(index)
    report["controls"] = available
    auto = resolve(AUTO_MODE_CANDIDATES, available)
    report["auto_control"] = auto
    if auto is None:
        report["error"] = (f"device {index} advertises no auto-exposure control; tried "
                           f"{list(AUTO_MODE_CANDIDATES)}, device offers {available}")
        return report
    report["auto_set"] = set_control(index, auto, AE_MANUAL)

    if exposure_abs is not None:
        exposure = resolve(EXPOSURE_ABS_CANDIDATES, available)
        report["exposure_control"] = exposure
        if exposure is None:
            report["error"] = (f"no absolute-exposure control; tried "
                               f"{list(EXPOSURE_ABS_CANDIDATES)}, device offers {available}")
            return report
        report["exposure_set"] = set_control(index, exposure, exposure_abs)
    return report


def lock_holds(index: int, report: dict) -> bool:
    """Read the mode back. Opening a device can silently reset it."""
    control = report.get("auto_control")
    if not control:
        return False
    value = get_control(index, control)
    if value is None:
        return False
    digits = "".join(c for c in value if c.isdigit())
    return bool(digits) and digits.startswith(str(AE_MANUAL))


def describe(report: dict) -> str:
    if report.get("error"):
        return f"exposure NOT locked: {report['error']}"
    parts = [f"manual via {report['auto_control']}"]
    if report.get("exposure_control"):
        parts.append(f"time via {report['exposure_control']}")
    return "exposure locked: " + ", ".join(parts)


def _selftest() -> None:
    print("[test] control-name resolution adapts to what the device advertises")
    assert resolve(AUTO_MODE_CANDIDATES, ["auto-exposure-mode", "gain"]) == "auto-exposure-mode"
    assert resolve(AUTO_MODE_CANDIDATES, ["ae-mode"]) == "ae-mode"
    assert resolve(AUTO_MODE_CANDIDATES, ["brightness"]) is None
    assert resolve(EXPOSURE_ABS_CANDIDATES, ["exposure-time-abs"]) == "exposure-time-abs"
    print("       ok")

    print("[test] parsing tolerates bullet and index prefixes across uvc-util builds")
    sample = "  * auto-exposure-mode", "  2) exposure-time-abs", "gain", "", "   "
    names = parse_controls("\n".join(sample))
    assert names == ["auto-exposure-mode", "exposure-time-abs", "gain"], names
    print("       ok")

    print("[test] an unsupported platform reports rather than silently no-opping")
    report = apply_lock(0, None)
    if sys.platform != "darwin":
        assert report["error"], report
        assert "no exposure-lock path" in report["error"], report
        assert not report["auto_set"]
        assert describe(report).startswith("exposure NOT locked")
    print("       ok")

    print("[test] a lock that was never established never reads back as held")
    assert not lock_holds(0, {"auto_control": None})
    assert not lock_holds(0, {})
    print("       ok")

    print("\nAll checks passed.")
    print(f"NOTE: uvc-util itself is UNEXERCISED - it is macOS-only and this ran on "
          f"{sys.platform}. Every subprocess call here is untested against real hardware.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        index = 0
        for i, arg in enumerate(sys.argv):
            if arg == "-I" and i + 1 < len(sys.argv):
                index = int(sys.argv[i + 1])
        print(f"platform supported : {supported()}  ({sys.platform})")
        print(f"{UVC_UTIL} on PATH   : {tool_available()}")
        if supported() and tool_available():
            print(f"controls on device {index}:")
            for name in list_controls(index) or ["  (none reported)"]:
                print(f"  {name}")
        else:
            print("Run this on the Mac with the camera attached to list its controls.")
