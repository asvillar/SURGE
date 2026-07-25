"""Model registry helpers."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

from .base import BaseModelAdapter


@dataclass
class RegisteredModel:
    key: str
    adapter_cls: Type[BaseModelAdapter]
    aliases: Tuple[str, ...] = ()


@dataclass
class RegistrationRecord:
    """Outcome of one adapter registration attempt.

    status is one of:
      registered — adapter is available in MODEL_REGISTRY
      skipped    — an optional dependency is missing/broken; reason says why
      error      — the adapter itself raised (a SURGE bug, never silent)
    """

    key: str
    status: str
    reason: str = ""
    requires: Tuple[str, ...] = field(default_factory=tuple)


REGISTRATION_LOG: List[RegistrationRecord] = []


def registration_report() -> List[RegistrationRecord]:
    """All registration attempts (registered / skipped / error) in order."""
    return list(REGISTRATION_LOG)


def registration_table() -> str:
    """Human-readable registration report for CLI/debug output."""
    if not REGISTRATION_LOG:
        return "No registration records."
    width = max(len(r.key) for r in REGISTRATION_LOG) + 2
    lines = []
    for rec in REGISTRATION_LOG:
        line = f"  {rec.key:<{width}}{rec.status}"
        if rec.reason:
            line += f"  — {rec.reason}"
        lines.append(line)
    n_reg = sum(1 for r in REGISTRATION_LOG if r.status == "registered")
    n_skip = sum(1 for r in REGISTRATION_LOG if r.status == "skipped")
    n_err = sum(1 for r in REGISTRATION_LOG if r.status == "error")
    header = (f"Adapter registration: {n_reg} registered, "
              f"{n_skip} skipped, {n_err} errors\n")
    return header + "\n".join(lines)


class ModelRegistry:
    """Registry capable of handling aliases and metadata."""

    def __init__(self) -> None:
        self._registry: Dict[str, RegisteredModel] = {}
        self._alias_map: Dict[str, str] = {}

    def register(
        self,
        key: str,
        adapter_cls: Type[BaseModelAdapter],
        aliases: Iterable[str] | None = None,
    ) -> RegisteredModel:
        aliases = tuple(dict.fromkeys(list(aliases or [])))
        registered = RegisteredModel(key=key, adapter_cls=adapter_cls, aliases=aliases)
        self._registry[key] = registered
        for alias in aliases:
            self._alias_map[alias] = key
        return registered

    def _resolve_key(self, name: str) -> str:
        if name in self._registry:
            return name
        if name in self._alias_map:
            return self._alias_map[name]
        raise KeyError(f"Model '{name}' is not registered")

    def get(self, name: str) -> RegisteredModel:
        return self._registry[self._resolve_key(name)]

    def create(self, name: str, **kwargs) -> BaseModelAdapter:
        registered = self.get(name)
        return registered.adapter_cls(**kwargs)

    def list_models(self) -> Dict[str, str]:
        return {key: value.adapter_cls.__name__ for key, value in sorted(self._registry.items())}

    def keys(self) -> List[str]:
        keys = list(self._registry.keys()) + list(self._alias_map.keys())
        return sorted(dict.fromkeys(keys))

    def __contains__(self, name: str) -> bool:
        return name in self._registry or name in self._alias_map


MODEL_REGISTRY = ModelRegistry()


def register_model(
    adapter_cls: Type[BaseModelAdapter],
    *,
    key: str | None = None,
    aliases: Iterable[str] | None = None,
) -> Type[BaseModelAdapter]:
    MODEL_REGISTRY.register(key or adapter_cls.name, adapter_cls, aliases=aliases)
    return adapter_cls


def get_model_class(name: str) -> Type[BaseModelAdapter]:
    return MODEL_REGISTRY.get(name).adapter_cls


def create_model(name: str, **kwargs) -> BaseModelAdapter:
    return MODEL_REGISTRY.create(name, **kwargs)


def list_models() -> Dict[str, str]:
    return MODEL_REGISTRY.list_models()
