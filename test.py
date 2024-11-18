import sys

from evolution.exagp import EXAGP
from evolution.autoencoder_node_generator import AutoencoderNodeGenerator
from evolution.autoencoder_edge_generator import AutoencoderEdgeGenerator

from genomes.minimal_recurrent_genome import MinimalRecurrentGenome
from genomes.trivial_recurrent_genome import TrivialRecurrentGenome
from genomes.autoencoder_genome import AutoencoderGenome
from genomes.autoencoder_genome2 import AutoencoderGenome2

from loguru import logger

from torch import optim

from time_series.time_series import TimeSeries

import pickle

if __name__ == "__main__":
    sys.setrecursionlimit(6000)
    logger.remove()
    logger.add(sys.stdout, level="INFO", backtrace=True, diagnose=True)

    csv_filename = (
        "/Users/aryanjha/Documents/exact/datasets/cats/cats_sample_30k.csv"
    )

    initial_series = TimeSeries.create_from_csv(filename=csv_filename)

    print(initial_series.series_dictionary)

    input_series_names = [
        "bed1", "bed2", "bfo1", "bfo2", "bso1", "bso2", "bso3", "ced1", "cfo1", "cso1",
    ]
    output_series_names = [
        "bed1", "bed2", "bfo1", "bfo2", "bso1", "bso2", "bso3", "ced1", "cfo1", "cso1",
    ]

    input_series = initial_series.get_inputs(
        input_series_names=input_series_names, offset=0
    )
    output_series = initial_series.get_outputs(
        output_series_names=output_series_names, offset=0
    )

    input_series = input_series.slice(0, 3000)
    output_series = output_series.slice(0, 3000)

    print(f"input_series -- n series: {len(input_series.series_dictionary)}")
    print(input_series.series_dictionary)
    print(f"output_series -- n series: {len(output_series.series_dictionary)}")
    print(output_series.series_dictionary)

    max_sequence_length = input_series.series_length
    print(f"max sequence length: {max_sequence_length}")

    seed_genome = MinimalRecurrentGenome(
        generation_number=0,
        input_series_names=input_series_names,
        output_series_names=output_series_names,
        max_sequence_length=max_sequence_length,
    )

    seed_genome = TrivialRecurrentGenome(
        generation_number=0,
        input_series_names=input_series_names,
        output_series_names=output_series_names,
        max_sequence_length=max_sequence_length,
    )
    exagp = EXAGP(seed_genome=seed_genome)

    seed_genome = AutoencoderGenome2(
        generation_number=0,
        input_series_names=input_series_names,
        output_series_names=output_series_names,
        max_sequence_length=max_sequence_length,
    )

    exagp = EXAGP(seed_genome=seed_genome, autoencoder=True)

    # seed_genome = AutoencoderGenome(
    #     generation_number=0,
    #     input_series_names=input_series_names,
    #     output_series_names=output_series_names,
    #     max_sequence_length=max_sequence_length,
    # )
    #
    # exagp = EXAGP(
    #     seed_genome=seed_genome,
    #     node_generator=AutoencoderNodeGenerator(),
    #     edge_generator=AutoencoderEdgeGenerator(max_time_skip=10)
    # )

    for genome_number in range(2000):
        new_genome = exagp.generate_genome()
        print(f"evaluating genome: {new_genome.generation_number}")
        optimizer = optim.Adam(new_genome.parameters(), lr=0.001)

        new_genome.train(
            input_series=input_series,
            output_series=output_series,
            optimizer=optimizer,
            iterations=10,
        )
        exagp.insert_genome(new_genome)

    print()
    print()

    best_fit_genome = exagp.population_strategy.population[0]
    print(f"{best_fit_genome}")

    # save best fit genome object
    pkl_filename = './test_genomes/genome_' + str(best_fit_genome.generation_number) + '.pkl'
    with open(pkl_filename, 'wb') as pkl_file:
        pickle.dump(best_fit_genome, pkl_file)

    best_fit_genome.plot()
