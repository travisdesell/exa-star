from typing import TYPE_CHECKING, Any, Dict, Self, Optional

if TYPE_CHECKING:
    import multiprocessing as mp
else:
    import multiprocess as mp


from config import configclass
from evolution import InitTask, InitTaskConfig, ParallelMPStrategy

from loguru import logger


class inon_t(int):
    """
    An innovation number. This is a subclass of int and should have no more overhead than an int and inherits all of its
    ordering, arithmetic, etc.
    """
    counter: int = 0
    divisor: int = 1

    @staticmethod
    def set_mod_class(mod_class: int, divisor: int) -> None:
        """
        Sets the congruence class that will be used in this process. Congruence classes are non-overlapping sets of
        integers `mod divisor`.
        https://en.wikipedia.org/wiki/Modular_arithmetic#Congruence_classes
        """
        inon_t.counter = mod_class
        inon_t.divisor = divisor

    def __new__(cls, value: Optional[int] = None, *args, **kwargs) -> Self:
        value = value if value is not None else inon_t._next()
        return int.__new__(cls, value)

    @staticmethod
    def _next() -> int:
        """
        Returns:
            The next unique innovation number.
        """
        number = inon_t.counter
        inon_t.counter += inon_t.divisor
        return number

    def __eq__(self, other) -> bool:
        """
        Override eq to allow subclasses of `inon_t` that won't be considered equal with one another.

        So, if we have `edge_inon_t(4)` and `node_inon_t(4)`, they should not be equal. This is simply done by checking
        the type. This is done by using an exact type comparison
        """
        return super().__eq__(other) and type(self) is type(other)

    def __hash__(self) -> int:
        return self


class InonInitTask[E: ParallelMPStrategy](InitTask):
    """
    An initialization task (see ParallelMPStrategy) that ensures that no two processes are using an overlapping set of
    innovation numbers.
    """

    def __init__(self) -> None:
        super().__init__()

    def run(self, values: Dict[str, Any]) -> None:
        """
        Fetch an integer from the id_queue (see `self.values` to see how this is created) and use it to set the modular
        congruence class of the innovation numbers for this process.
        """
        mod_class, divisor = values["id_queue"].get()
        inon_t.set_mod_class(mod_class, divisor)
        logger.info(f"setting mod class to `{mod_class} mod {divisor}`")

    def values(self, strategy: E) -> Dict[str, Any]:
        """
        Creates a queue and places integers [0, strategy.parallelism + 1) in the queue. The innovation number space will
        effectively be partitioned into strategy.parallelism + 1 non-overlapping sets.
        """
        q = mp.Queue()

        for i in range(strategy.parallelism):
            q.put((i, strategy.parallelism + 1))

        inon_t.set_mod_class(strategy.parallelism, strategy.parallelism + 1)

        return {"id_queue": q}


@configclass(name="base_inon_init_task", group="init_tasks", target=InonInitTask)
class InonInitTaskConfig(InitTaskConfig):
    ...
