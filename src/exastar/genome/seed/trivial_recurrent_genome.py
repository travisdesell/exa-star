from config import configclass
from exastar.genome.seed.seed_genome_factory import SeedGenomeFactory, SeedGenomeFactoryConfig
from exastar.genome.recurrent_genome import RecurrentGenome
from exastar.time_series import TimeSeries
from exastar.weights import WeightGenerator

import numpy as np


class TrivialRecurrentGenomeFactory(SeedGenomeFactory[RecurrentGenome]):
    def __call__(
        self,
        dataset: TimeSeries,
        weight_generator: WeightGenerator,
        rng: np.random.Generator
    ) -> RecurrentGenome:
        return RecurrentGenome.make_trivial(
            0, dataset.input_series_names, dataset.output_series_names, dataset.series_length, weight_generator, rng
        )


@configclass(name="base_trivial_recurrent_seed_genome_factory", group="genome_factory/seed_genome_factory",
             target=TrivialRecurrentGenomeFactory)
class TrivialRecurrentSeedGenomeFactoryConfig(SeedGenomeFactoryConfig):
    ...
