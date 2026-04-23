"""SoundCard 用の NumPy 互換パッチ（廃止された ``fromstring`` の代替）。"""

from __future__ import annotations

import numpy as np

_PATCHED = False


def apply_numpy_patch() -> None:
    """SoundCard が期待する ``numpy.fromstring`` 互換実装を一度だけ適用する。

    バイナリバッファからの配列生成は ``frombuffer`` に委譲する。

    Returns:
        None
    """
    global _PATCHED
    if _PATCHED or getattr(np, "_soundcard_fromstring_patched", False):
        return

    _orig = getattr(np, "fromstring", None)

    def fromstring(
        string,
        dtype=float,
        count=-1,
        sep="",
        *,
        like=None,
    ):
        """``numpy.fromstring`` 互換。``sep==""`` は ``frombuffer`` 経由。"""
        if sep != "":
            if _orig is not None:
                return _orig(string, dtype=dtype, count=count, sep=sep, like=like)
            raise RuntimeError("numpy.fromstring(..., sep=...) is not available")
        mv = memoryview(string)
        if count is None or count < 0:
            arr = np.frombuffer(mv, dtype=dtype)
        else:
            arr = np.frombuffer(mv, dtype=dtype, count=int(count))
        return arr.copy()

    np.fromstring = fromstring  # type: ignore[method-assign]
    np._soundcard_fromstring_patched = True
    _PATCHED = True
