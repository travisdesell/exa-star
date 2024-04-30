from abc import ABC, abstractmethod
from functools import total_ordering
from typing import Any, Dict, Tuple


@total_ordering
class ComparableMixin(ABC):

    @abstractmethod
    def _cmpkey(self) -> Tuple: ...

    def _compare(self, other, f):
        if type(other) != type(self):
            raise TypeError(
                f"ComparableMixing comparisons must be called with identical types."
            )

        a, b = self._cmpkey(), other._cmpkey()

        if len(a) != len(b):
            raise TypeError(f"cmp key must have fixed length")

        return f(self._cmpkey(), other._cmpkey())

    def __lt__(self, other):
        return self._compare(other, lambda a, b: a < b)

    def __eq__(self, other):
        return self._compare(other, lambda a, b: a == b)


def constmethod(func):
    """
    Decorator to make a method const. An exception will the thrown if an attribute change is attempted.
    """

    def wrapper(instance):
        class unsettable_class(instance.__class__):
            def __init__(self):
                super().__setattr__("__dict__", instance.__dict__)

            def __setattr__(self, attr, value):
                if hasattr(self, attr):
                    raise AttributeError(
                        f"Trying to set value: {value} on the read-only attribute: {instance.__class__.__name__}.{attr}"
                    )
                super().__setattr__(attr, value)

        return func(unsettable_class())

    return wrapper


class LogDataProvider[T]:

    def __init__(self) -> None: ...

    def prefix(self, prefix: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._modify_keys(prefix, "", data)

    def suffix(self, suffix: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._modify_keys("", suffix, data)

    def _modify_keys(
        self, prefix: str, suffix: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {f"{prefix}{k}{suffix}": v for k, v in data.items()}

    @abstractmethod
    def get_log_data(self, aggregator: T) -> Dict[str, Any]: ...


class LogDataAggregator[T](LogDataProvider):

    def __init__(self, providers: Dict[str, LogDataProvider]) -> None:
        super().__init__()

        self._providers: Dict[str, LogDataProvider] = providers

    def get_log_data(self, aggregator: T) -> Dict[str, Any]:
        data: Dict[str, Any] = {}

        def append_data(new_data: Dict[str, Any]):
            intersection = new_data.keys() & data.keys()
            if intersection:
                raise ValueError(f"Log provider gave duplicate key(s): {intersection}")

            data.update(new_data)

        for _, value in self.__dict__.items():
            if not isinstance(value, LogDataProvider):
                continue

            append_data(value.get_log_data(self))

        for _, provider in self._providers.items():
            append_data(provider.get_log_data(self))

        return data
