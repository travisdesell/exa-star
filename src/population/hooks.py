from dataclasses import dataclass

from loguru import logger

from config import configclass
from genome import Genome
from util.hook import Hook, HookConfig


class GenomeInsertedHook[G: Genome](Hook[G]):
    ...


@dataclass
class GenomeInsertedHookConfig(HookConfig):
    ...


class GenomeRemovedHook[G: Genome](Hook[G]):
    ...


@dataclass
class GenomeRemovedHookConfig[G: Genome](Hook[G]):
    ...


class PrintInsertedGenomeHook[G: Genome](GenomeInsertedHook[G]):

    def on_event(self, value: G) -> None:
        logger.info(f"Inserted genome: {value}")


@configclass(name="base_print_inserted_genome", group="hook", target=PrintInsertedGenomeHook)
class PrintInsertedGenomeHookConfig(GenomeInsertedHookConfig):
    ...
