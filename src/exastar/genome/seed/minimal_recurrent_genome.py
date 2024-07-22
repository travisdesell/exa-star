from config import configclass
from exastar.genome.seed.seed_genome_factory import SeedGenomeFactory, SeedGenomeFactoryConfig
from exastar.genome.recurrent_genome import RecurrentGenome
from exastar.time_series import TimeSeries


class MinimalRecurrentGenomeFactory(SeedGenomeFactory[RecurrentGenome]):

    def __call__(self, dataset: TimeSeries) -> RecurrentGenome:
        return RecurrentGenome.make_minimal_recurrent(
            0, dataset.input_series_names, dataset.output_series_names, dataset.series_length
        )


@configclass(name="base_minimal_recurrent_seed_genome", group="genome_factory/seed_genome_factory",
             target=MinimalRecurrentGenomeFactory)
class MinimalRecurrentSeedGenomeFactoryConfig(SeedGenomeFactoryConfig):
    ...
