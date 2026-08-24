"""Import motmetrics under NumPy 2.

motmetrics 1.4.0 (the current release) calls np.asfarray, which NumPy removed in 2.0:

    AttributeError: `np.asfarray` was removed in the NumPy 2.0 release.

This project is pinned to numpy 2.5.2 and cannot move off it - torch, torchvision and
Ultralytics are all built against NumPy 2, and the entire detection sweep was trained
and benchmarked on it. Downgrading NumPy to suit one metrics library would invalidate
work that is already finished.

The damage is small enough to patch precisely rather than work around. Only two call
sites in the shipped library are affected, both in motmetrics/distances.py:

    objs = np.asfarray(objs)
    hyps = np.asfarray(hyps)

Every other occurrence is under motmetrics/tests/, which is never executed here.
np.asfarray(a) was defined as np.asarray(a, dtype=np.float64), so the replacement below
is the historical behaviour rather than an approximation.

Alternatives considered: computing distance matrices ourselves and feeding
MOTAccumulator directly, which avoids the broken path but reimplements MOT's IoU
conventions - more code and more room to diverge from the standard than restoring one
removed alias. If motmetrics ever ships a NumPy 2 release, delete this module and import
motmetrics directly.

Usage:
    from mot_compat import mm      # patched motmetrics, ready to use
"""

from __future__ import annotations

import numpy as np

if not hasattr(np, "asfarray"):
    def _asfarray(a, dtype=np.float64):
        """NumPy's removed asfarray: asarray with a float dtype."""
        return np.asarray(a, dtype=dtype)

    np.asfarray = _asfarray  # type: ignore[attr-defined]

import motmetrics as mm  # noqa: E402 - must follow the shim

__all__ = ["mm"]
