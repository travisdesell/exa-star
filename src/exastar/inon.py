import multiprocess as mp
import threading
from typing import Any, Dict, Self

from config import configclass
from evolution import InitTask, InitTaskConfig, SynchronousMTStrategy


class inon_t(int):
    counter: int = 0
    mod_class: int = 0
    divisor: int = 1

    @staticmethod
    def set_mod_class(mod_class: int, divisor: int) -> None:
        inon_t.counter = mod_class
        inon_t.divisor = divisor

    def __new__(cls, *args, **kwargs) -> Self:
        return int.__new__(cls, inon_t._next())

    @staticmethod
    def _next() -> int:
        """
        Returns:
            The next unique innovation number.
        """
        number = inon_t.counter
        inon_t.counter += inon_t.divisor
        return number


class InonInitTask(InitTask):

    def __init__(self) -> None:
        super().__init__()

    def run(self, values: Dict[str, Any]) -> None:
        mod_class, divisor = values["id_queue"].get()
        inon_t.set_mod_class(mod_class, divisor)
        print(f"mod_class {mod_class}, divisor {divisor}")

    def values(self, strategy: SynchronousMTStrategy) -> Dict[str, Any]:
        q = mp.Queue()

        for i in range(strategy.parallelism):
            q.put((i, strategy.parallelism + 1))

        inon_t.set_mod_class(strategy.parallelism, strategy.parallelism + 1)

        return {"id_queue": q}


@configclass(name="base_inon_init_task", group="init_tasks", target=InonInitTask)
class InonInitTaskConfig(InitTaskConfig):
    ...
