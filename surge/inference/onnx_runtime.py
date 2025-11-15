"""ONNX Runtime helpers for SURGE inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    ONNX_AVAILABLE = False
    ort = None  # type: ignore


def load_session(model_path: str | Path) -> Any:
    """Load an ONNX model if onnxruntime is installed."""
    if not ONNX_AVAILABLE:
        raise ImportError("onnxruntime is required for ONNX inference. pip install onnxruntime")
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), providers=providers)


def infer(session: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference on the provided ONNX session."""
    if not ONNX_AVAILABLE:
        raise ImportError("onnxruntime is required for ONNX inference.")
    return {
        output.name: session.run([output.name], inputs)[0]
        for output in session.get_outputs()
    }
