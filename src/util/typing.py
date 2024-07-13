from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, Type


class ComparableMixin:
    """
    If you are using multiple inheritence, it is pertinent that this class is inherited before any interfaces
    or abstract classes. Failure to do so will mess up the order of constructor calls.
    """

    def __init__(self, type: Optional[Type] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._comparison_parent_type: Optional[Type] = type

    @abstractmethod
    def _cmpkey(self) -> Tuple: ...

    def _compare(self, other, f):
        if self._comparison_parent_type \
            and (not isinstance(self, self._comparison_parent_type)
                 or not isinstance(other, self._comparison_parent_type)):
            raise TypeError(
                f"ComparableMixing comparisons must match the self._comparison_parent_type = "
                f"{self._comparison_parent_type}"
            )

        a, b = self._cmpkey(), other._cmpkey()

        if len(a) != len(b):
            raise TypeError("_cmpkey must have fixed length")

        return f(self._cmpkey(), other._cmpkey())

    def __lt__(self, other):
        return self._compare(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._compare(other, lambda a, b: a <= b)

    def __gt__(self, other):
        return self._compare(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._compare(other, lambda a, b: a >= b)

    def __eq__(self, other):
        return self._compare(other, lambda a, b: a == b)

    def __ne__(self, other):
        return self._compare(other, lambda a, b: a != b)


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


def overrides(interface_class):
    def overrider(method):
        assert (method.__name__ in dir(interface_class))
        return method
    return overrider


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
    @constmethod
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
