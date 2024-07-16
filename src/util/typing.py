from abc import abstractmethod
from typing import Optional, Tuple, Type


class ComparableMixin:
    """
    If you are using multiple inheritence, it is pertinent that this class is inherited before any interfaces
    or abstract classes. Failure to do so will mess up the order of constructor calls.
    """

    def __init__(self, type: Optional[Type] = None, *args, **kwargs) -> None:
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
