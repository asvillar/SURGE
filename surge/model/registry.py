"""Model registry helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Type

from .base import BaseModelAdapter


@dataclass
class RegisteredModel:
    key: str
    adapter_cls: Type[BaseModelAdapter]
    aliases: Tuple[str, ...] = ()


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
