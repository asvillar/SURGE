"""Hidden-layer width schedules for tabular PyTorch MLPs."""

from __future__ import annotations

import math
from typing import Literal

LayerSchedule = Literal["explicit", "geometric"]


def clamp_layer_widths(
    widths: list[int],
    *,
    min_width: int = 1,
    max_width: int = 1024,
) -> list[int]:
    """Clamp each hidden width to ``[min_width, max_width]``."""
    lo = max(1, int(min_width))
    hi = max(lo, int(max_width))
    return [max(lo, min(hi, int(w))) for w in widths]


def geometric_hidden_widths(
    n_in: int,
    n_out: int,
    n_hidden: int,
    *,
    min_width: int = 1,
    max_width: int = 1024,
) -> list[int]:
    """Widths along a geometric path from input to output dimension.

    Treat the network as ``n_in → h₁ → … → hₖ → n_out`` with ``k = n_hidden``
    hidden layers and a constant ratio ``r`` between consecutive sizes:

        hᵢ = n_in · r^i,   n_out = n_in · r^(k+1)  ⇒  r = (n_out / n_in)^(1/(k+1))

    Each width is rounded and clamped to ``[min_width, max_width]``.
    """
    k = max(0, int(n_hidden))
    if k == 0:
        return []
    n_in = max(1, int(n_in))
    n_out = max(1, int(n_out))
    ratio = (n_out / n_in) ** (1.0 / (k + 1))
    raw = [n_in * (ratio ** i) for i in range(1, k + 1)]
    return clamp_layer_widths(
        [int(round(w)) for w in raw],
        min_width=min_width,
        max_width=max_width,
    )


def resolve_hidden_layers(
    *,
    n_in: int,
    n_out: int,
    schedule: LayerSchedule = "explicit",
    hidden_layers: list[int] | None = None,
    n_hidden_layers: int | None = None,
    min_width: int = 1,
    max_width: int = 1024,
) -> list[int]:
    """Return hidden layer widths for the chosen schedule."""
    if schedule == "geometric":
        n_h = n_hidden_layers if n_hidden_layers is not None else 2
        return geometric_hidden_widths(
            n_in, n_out, n_h, min_width=min_width, max_width=max_width
        )
    layers = list(hidden_layers or [128, 128])
    return clamp_layer_widths(layers, min_width=min_width, max_width=max_width)
