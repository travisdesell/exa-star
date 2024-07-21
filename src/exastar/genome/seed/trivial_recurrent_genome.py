from config import configclass
from exastar.genome.seed.seed_genome_factory import SeedGenomeFactory, SeedGenomeFactoryConfig
from exastar.genome.recurrent_genome import RecurrentGenome
from exastar.time_series import TimeSeries


class TrivialRecurrentGenomeFactory(SeedGenomeFactory[RecurrentGenome]):

    def __call__(self, dataset: TimeSeries) -> RecurrentGenome:
        return RecurrentGenome.make_trivial(
            0, dataset.input_series_names, dataset.output_series_names, dataset.series_length
        )


@configclass(name="base_trivial_recurrent_seed_genome_factory", group="genome_factory/seed_genome_factory",
             target=TrivialRecurrentGenomeFactory)
class TrivialRecurrentSeedGenomeFactoryConfig(SeedGenomeFactoryConfig):
    ...
