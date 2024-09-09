from typing import TYPE_CHECKING, Any, Dict, Self, Optional

if TYPE_CHECKING:
    import multiprocessing as mp
else:
    import multiprocess as mp


from config import configclass
from evolution import InitTask, InitTaskConfig, ParallelMPStrategy

from loguru import logger


class inon_t(int):
    counter: int = 0
    mod_class: int = 0
    divisor: int = 1

    @staticmethod
    def set_mod_class(mod_class: int, divisor: int) -> None:
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


class InonInitTask[E: ParallelMPStrategy](InitTask):

    def __init__(self) -> None:
        super().__init__()

    def run(self, values: Dict[str, Any]) -> None:
        mod_class, divisor = values["id_queue"].get()
        inon_t.set_mod_class(mod_class, divisor)
        logger.info(f"setting mod class to `{mod_class} mod {divisor}`")

    def values(self, strategy: E) -> Dict[str, Any]:
        q = mp.Queue()

        for i in range(strategy.parallelism):
            q.put((i, strategy.parallelism + 1))

        inon_t.set_mod_class(strategy.parallelism, strategy.parallelism + 1)

        return {"id_queue": q}


@configclass(name="base_inon_init_task", group="init_tasks", target=InonInitTask)
class InonInitTaskConfig(InitTaskConfig):
    ...
