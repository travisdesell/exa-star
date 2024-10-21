from typing import Any, Dict

from config import configclass
from dataset import Dataset
from genome import Genome
from population.population import Population
from population.steady_state_population import SteadyStatePopulation
from util.log import LogDataProvider, LogDataProviderConfig


class LogBestGenome[G: Genome, D: Dataset](LogDataProvider[Population[G, D]]):

    def get_log_data(self, aggregator: Population[G, D]) -> Dict[str, Any]:
        """
        Grab the best genomes' log data from the population.
        """
        return self.prefix(
            "best_genome_", aggregator.get_best_genome().get_log_data(None)
        )


@configclass(name="base_log_best_genome", group="log_data_providers", target=LogBestGenome)
class LogBestGenomeConfig(LogDataProviderConfig):
    ...


class LogWorstGenome[G: Genome, D: Dataset](LogDataProvider[Population[G, D]]):

    def get_log_data(self, aggregator: Population[G, D]) -> Dict[str, Any]:
        """
        Grab the worst best genomes' log data from the population.
        """
        return self.prefix(
            "worst_genome_", aggregator.get_worst_genome().get_log_data(None)
        )


@configclass(name="base_log_worst_genome", group="log_data_providers", target=LogWorstGenome)
class LogWorstGenomeConfig(LogDataProviderConfig):
    ...


class LogRecentGenome[G: Genome, D: Dataset](LogDataProvider[SteadyStatePopulation[G, D]]):

    def get_log_data(self, aggregator: SteadyStatePopulation[G, D]) -> Dict[str, Any]:
        """
        Grabs the log data for the most recent genome, only applicable to SteadStateGenome
        """
        return self.prefix(
            "recent_genome_", aggregator.most_recent_genome.get_log_data(None) if aggregator.most_recent_genome else {}
        )


@configclass(name="base_log_recent_genome", group="log_data_providers", target=LogRecentGenome)
class LogRecentGenomeConfig(LogDataProviderConfig):
    ...
