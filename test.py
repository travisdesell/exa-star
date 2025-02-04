import sys

from evolution.exagp import EXAGP
from evolution.bidirectionalAE_node_generator import BidirectionalAENodeGenerator
from evolution.bidirectionalAE_edge_generator import BidirectionalAEEdgeGenerator

from genomes.minimal_recurrent_genome import MinimalRecurrentGenome
from genomes.trivial_recurrent_genome import TrivialRecurrentGenome
from genomes.bidirectionalAE_genome import BidirectionalAEGenome
from genomes.autoencoder_genome import AutoencoderGenome

from loguru import logger

from torch import optim

from time_series.time_series import TimeSeries

import pickle

if __name__ == "__main__":
    sys.setrecursionlimit(6000)
    logger.remove()
    logger.add(sys.stdout, level="INFO", backtrace=True, diagnose=True)

    csv_filename = (
        "/Users/aryanjha/Documents/exact/datasets/smap-msl/smap/p_smap.csv"
    )

    initial_series = TimeSeries.create_from_csv(filename=csv_filename)

    print(initial_series.series_dictionary)

    input_series_names = [
        "telemetry_1","telemetry_2","telemetry_3","telemetry_4","telemetry_5","telemetry_6","telemetry_7","telemetry_8","telemetry_9","telemetry_10","telemetry_12","telemetry_13","telemetry_14","telemetry_15","telemetry_18","telemetry_19","telemetry_20","telemetry_22","telemetry_23"
    ]
    output_series_names = [
        "telemetry_1","telemetry_2","telemetry_3","telemetry_4","telemetry_5","telemetry_6","telemetry_7","telemetry_8","telemetry_9","telemetry_10","telemetry_12","telemetry_13","telemetry_14","telemetry_15","telemetry_18","telemetry_19","telemetry_20","telemetry_22","telemetry_23"
    ]

    input_series = initial_series.get_inputs(
        input_series_names=input_series_names, offset=0
    )
    output_series = initial_series.get_outputs(
        output_series_names=output_series_names, offset=0
    )

    # input_series = input_series.slice(0, 3000)
    # output_series = output_series.slice(0, 3000)

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

    seed_genome = AutoencoderGenome(
        generation_number=0,
        input_series_names=input_series_names,
        output_series_names=output_series_names,
        max_sequence_length=max_sequence_length,
    )

    exagp = EXAGP(seed_genome=seed_genome, autoencoder=True)

    # seed_genome = BidirectionalAEGenome(
    #     generation_number=0,
    #     input_series_names=input_series_names,
    #     output_series_names=output_series_names,
    #     max_sequence_length=max_sequence_length,
    # )
    #
    # exagp = EXAGP(
    #     seed_genome=seed_genome,
    #     node_generator=BidirectionalAENodeGenerator(),
    #     edge_generator=BidirectionalAEEdgeGenerator(max_time_skip=10)
    # )

    for genome_number in range(1000):
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
