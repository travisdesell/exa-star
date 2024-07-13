import threading
from typing import Self


class inon_t(int):
    counter: threading.local = threading.local()
    mod_class: int = 0
    divisor: int = 1

    @staticmethod
    def set_mod_class(mod_class: int, divisor: int) -> None:
        inon_t.counter.value = mod_class
        inon_t.divisor = divisor

    def __new__(cls, *args, **kwargs) -> Self:
        return int.__new__(cls, inon_t._next())

    @staticmethod
    def _next() -> int:
        """
        Returns:
            The next unique innovation number.
        """
        number = inon_t.counter.value
        inon_t.counter.value += inon_t.divisor
        return number
