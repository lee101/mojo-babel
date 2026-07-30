"""ctypes bindings for the Mojo formatting kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.environ.get(
    "MOJO_BABEL_LIB", os.path.join(ROOT, "dist", "libmojo-babel.so")
)

I = ctypes.c_int64

_SIGNATURES = {
    "mb_format_scaled": ([I] * 15, I),
    "mb_format_dates": ([I] * 13, I),
    "mb_decompose_days": ([I, I, I], I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.path.exists(LIB_PATH) and not force:
        return LIB_PATH
    if os.environ.get("MOJO_BABEL_LIB"):
        raise BuildError(f"MOJO_BABEL_LIB does not exist: {LIB_PATH}")
    pixi = shutil.which("pixi")
    command = (
        [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "build"]
        if pixi
        else ["bash", os.path.join(ROOT, "build", "build.sh")]
    )
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB_PATH):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB_PATH


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("native buffers must be NumPy arrays")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError("native buffers must be aligned and C-contiguous")
    return int(array.ctypes.data)


def bytes_array(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8)
