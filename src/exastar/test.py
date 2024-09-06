"""
import sys
from typing import Callable, List, Optional, Self

from config import configclass
from dataset import Dataset
from evolution import EvolutionaryStrategy, EvolutionaryStrategyConfig
from exastar.time_series import TimeSeries
from exastar.genome import EXAStarGenome

from loguru import logger
import numpy as np
from torch import optim

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", backtrace=True, diagnose=True)

    csv_filename = (
        "/Users/travisdesell/Data/stocks/most_newest/DJI_company_2023/train/AAPL.csv"
    )

    input_series_names = [
        "RET",
        "VOL_CHANGE",
        "BA_SPREAD",
        "ILLIQUIDITY",
        "sprtrn",
        "TURNOVER",
        "DJI_Return",
        "PRC",
    ]
    output_series_names = ["RET"]
    initial_series = TimeSeries.create_from_csv(
        filenames=list(csv_filename), input_series=input_series_names, output_series=output_series_names)

    print(initial_series.series_dictionary)
    input_series = initial_series.get_inputs(
        input_series_names=input_series_names, offset=1
    )
    output_series = initial_series.get_outputs(
        output_series_names=output_series_names, offset=1
    )

    input_series = input_series.slice(0, 50)
    output_series = output_series.slice(0, 50)

    print(f"input_series -- n series: {len(input_series.series_dictionary)}")
    print(input_series.series_dictionary)
    print(f"output_series -- n series: {len(output_series.series_dictionary)}")
    print(output_series.series_dictionary)

    max_sequence_length = input_series.series_length
    print(f"max sequence length: {max_sequence_length}")

    for genome_number in range(5000):
        new_genome = exagp.generate_genome()
        print(f"evaluating genome: {new_genome.generation_number}")
        optimizer = optim.Adam(new_genome.parameters(), lr=0.001)

        new_genome.train(
            input_series=input_series,
            output_series=output_series,
            optimizer=optimizer,
            iterations=0,
        )
        exagp.insert_genome(new_genome)

    print()
    print()
    print(f"{exagp.population_strategy.population[0]}")

    exagp.population_strategy.population[0].plot()


class EXAStarTestStrategy[G: EXAStarGenome, D: TimeSeries](EvolutionaryStrategy[G, D]):
    rng: np.random.Generator = np.random.default_rng()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.i: int = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        ...

    def step(self) -> None:
        logger.info("Starting step...")
        tasks: List[Callable[[np.random.Generator], Optional[G]]] = (
            self.population.make_generation(self.genome_factory)
        )

        genomes: List[G] = []

        for task in tasks:
            g = task(EXAStarTestStrategy.rng)
            if g:
                g.evaluate(self.fitness, self.dataset)

            return g

        self.population.integrate_generation(genomes)
        logger.info("step complete...")


@configclass(name="base_exastar_test_strategy", target=EXAStarTestStrategy)
class EXAStarTestStrategyConfig(EvolutionaryStrategyConfig):
    ...
"""
