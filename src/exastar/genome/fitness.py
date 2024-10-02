from typing import Dict, Tuple

from config import configclass
from genome import Fitness, FitnessConfig, FitnessValue, MSEValue
from exastar.genome.exastar_genome import EXAStarGenome
from exastar.time_series import TimeSeries

import torch


class EXAStarTimeSeriesRegressionFitness[G: EXAStarGenome](Fitness[G, TimeSeries]):

    def __init__(self) -> None:
        super().__init__()


@configclass(name="base_exastar_time_series_regression_fitness", group="fitness",
             target=EXAStarTimeSeriesRegressionFitness)
class EXAStarFitnessConfig(FitnessConfig):
    ...


class EXAStarMSE(EXAStarTimeSeriesRegressionFitness[EXAStarGenome]):

    def __init__(self) -> None:
        super().__init__()

    def compute(self, genome: EXAStarGenome, dataset: TimeSeries) -> MSEValue[EXAStarGenome]:
        value = MSEValue(
            genome.train_genome(dataset.get_inputs(dataset.input_series_names, 0),
                                dataset.get_outputs(dataset.output_series_names, 1),
                                torch.optim.Adam(genome.parameters()), 2)
        )
        return value


@ configclass(name="base_exastar_mse", group="fitness", target=EXAStarMSE)
class EXAStarMSEConfig(EXAStarFitnessConfig):
    ...
