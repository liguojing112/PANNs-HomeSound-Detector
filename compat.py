"""Compatibility helpers for third-party libraries with limited Python support."""

from __future__ import annotations

import sys
import types


def install_numba_stub() -> None:
    """Install a tiny numba shim when the real package cannot be imported."""
    try:
        import numba  # noqa: F401

        return
    except Exception:
        pass

    stub = types.ModuleType("numba")

    def _decorator(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def wrapper(func):
            return func

        return wrapper

    stub.jit = _decorator
    stub.njit = _decorator
    stub.vectorize = _decorator
    stub.guvectorize = _decorator
    stub.generated_jit = _decorator
    stub.stencil = _decorator
    stub.prange = range
    stub.config = types.SimpleNamespace(DISABLE_JIT=True)

    sys.modules["numba"] = stub
