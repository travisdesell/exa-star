from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, List, cast, Type


class Hook[T]:

    @staticmethod
    def filter[R: Hook](hooks: Iterable[Hook[Any]], type: Type[R]) -> List[R]:
        return cast(List[R], [h for h in hooks if isinstance(h, type)])

    def __init__(self) -> None:
        ...

    @abstractmethod
    def on_event(self, value: T) -> None:
        ...

    def __call__(self, value: T) -> None:
        self.on_event(value)


@dataclass
class HookConfig:
    ...
