from config import configclass
from genome import Fitness, FitnessConfig, MSEValue
from exastar.genome.exastar_genome import EXAStarGenome
from exastar.time_series import TimeSeries


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
        return MSEValue(4)


@configclass(name="base_exastar_mse", group="fitness", target=EXAStarMSE)
class EXAStarMSEConfig(EXAStarFitnessConfig):
    ...