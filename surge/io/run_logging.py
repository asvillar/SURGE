"""Stdout/stderr tee helpers for capturing workflow logs to disk."""

from __future__ import annotations

from typing import TextIO


class TeeText:
    """Mirror writes to multiple text streams (console + buffer or log file)."""

    def __init__(self, *streams: TextIO):
        self._streams = tuple(s for s in streams if s is not None)

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
            try:
                s.flush()
            except Exception:
                pass
        return len(data) if data else 0

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False
