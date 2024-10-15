from abc import abstractmethod
from dataclasses import dataclass
import os
from typing import Any, Dict

from config import configclass
from dataset import Dataset
from evolution.evolutionary_strategy import EvolutionaryStrategy
from genome import Genome

from loguru import logger


class InitTask[E: EvolutionaryStrategy]:
    """
    A task to be ran on a separate process to configure state / environment properly.
    """

    def __init__(self) -> None:
        ...

    @abstractmethod
    def run(self, values: Dict[str, Any]) -> None:
        """
        Run the initialization task - called once per process.
        """
        ...

    @abstractmethod
    def values(self, strategy: E) -> Dict[str, Any]:
        """
        Gathers any values from the strategy that may be necessary for the initialization task. These will be accessible
        with the `values` arguments in the `run` method.
        """
        ...


@dataclass
class InitTaskConfig:
    ...


class DatasetInitTask[E: EvolutionaryStrategy](InitTask[E]):
    """
    An initialization task to create a single, unique copy of the dataset on each process.
    """

    def __init__(self) -> None:
        super().__init__()

    def run(self, values: Dict[str, Any]) -> None:
        """
        Simply sets the static dataset value in `EvolutionaryStrategy`.
        """
        EvolutionaryStrategy.set_dataset(values["dataset"])

    def values[G: Genome, D: Dataset](self, strategy: EvolutionaryStrategy[G, D]) -> Dict[str, Any]:
        return {"dataset": strategy.dataset}


@configclass(name="base_dataset_init_task", group="init_tasks", target=DatasetInitTask)
class DatasetInitTaskConfig(InitTaskConfig):
    ...


class EnvironmentInitTask[E: EvolutionaryStrategy](InitTask[E]):
    """
    Sets environment variables (i.e. `os.environ`).

    NOTE: Some environment variables are read at the time of process creation, and changing them here will NOT change
    behavior. You will have to ensure your environmental variables to not fall into that category to rely on this.
    """

    def __init__(self, environment: Dict[str, str]) -> None:
        super().__init__()
        self.environment: Dict[str, str] = environment

    def run(self, values: Dict[str, Any]) -> None:
        logger.info("RUNNING TASK")
        for name, value in self.environment.items():
            if name in os.environ:
                logger.info(f"Overwriting key {name}:{os.environ[name]} to {value}")
            else:
                logger.info(f"Writing to environment {name}:{value}")
            os.environ[name] = value

    def values(self, strategy: E) -> Dict[str, Any]:
        return {}


@configclass(name="base_environment_init_task", group="init_tasks", target=EnvironmentInitTask)
class EnvironmentInitTaskConfig(InitTaskConfig):
    environment: Dict[str, str]
