import sys
import configparser
from loguru import logger
from torch import optim
import pickle

from evolution.exagp import EXAGP
from evolution.bidirectionalAE_node_generator import BidirectionalAENodeGenerator
from evolution.bidirectionalAE_edge_generator import BidirectionalAEEdgeGenerator

from genomes.minimal_recurrent_genome import MinimalRecurrentGenome
from genomes.trivial_recurrent_genome import TrivialRecurrentGenome
from genomes.bidirectionalAE_genome import BidirectionalAEGenome
from genomes.autoencoder_genome import AutoencoderGenome

from time_series.time_series import TimeSeries

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", backtrace=True, diagnose=True)

    config = configparser.ConfigParser()
    config.read('exa_ae_config.ini')

    csv_filename = config['DEFAULT']['training_data']
    input_series_names = config.get('DEFAULT', 'parameters').split(',')
    output_series_names = config.get('DEFAULT', 'parameters').split(',')
    bidirectional_ae = config.getboolean('DEFAULT', 'bidirectional_ae')
    num_generations = config.getint('DEFAULT', 'num_generations')
    num_iterations = config.getint('DEFAULT', 'num_iterations')
    learning_rate = config.getfloat('DEFAULT', 'learning_rate')

    initial_series = TimeSeries.create_from_csv(filename=csv_filename)
    print(initial_series.series_dictionary)


    input_series = initial_series.get_inputs(
        input_series_names=input_series_names, offset=0
    )
    output_series = initial_series.get_outputs(
        output_series_names=output_series_names, offset=0
    )

    print(f"input_series -- n series: {len(input_series.series_dictionary)}")
    print(input_series.series_dictionary)
    print(f"output_series -- n series: {len(output_series.series_dictionary)}")
    print(output_series.series_dictionary)

    max_sequence_length = input_series.series_length
    print(f"max sequence length: {max_sequence_length}")

    if not bidirectional_ae:
        seed_genome = AutoencoderGenome(
            generation_number=0,
            input_series_names=input_series_names,
            output_series_names=output_series_names,
            max_sequence_length=max_sequence_length,
        )

        exagp = EXAGP(seed_genome=seed_genome, autoencoder=True)

    else:
        seed_genome = BidirectionalAEGenome(
            generation_number=0,
            input_series_names=input_series_names,
            output_series_names=output_series_names,
            max_sequence_length=max_sequence_length,
        )

        exagp = EXAGP(
            seed_genome=seed_genome,
            node_generator=BidirectionalAENodeGenerator(),
            edge_generator=BidirectionalAEEdgeGenerator(max_time_skip=10)
        )

    for genome_number in range(num_generations):
        new_genome = exagp.generate_genome()
        print(f"evaluating genome: {new_genome.generation_number}")
        optimizer = optim.Adam(new_genome.parameters(), lr=learning_rate)

        new_genome.train(
            input_series=input_series,
            output_series=output_series,
            optimizer=optimizer,
            iterations=num_iterations,
        )
        exagp.insert_genome(new_genome)

    print()
    print()

    best_fit_genome = exagp.population_strategy.population[0]
    print(f"{best_fit_genome}")

    # save the best-fit genome object
    pkl_filename = './test_genomes/genome_' + str(best_fit_genome.generation_number) + '.pkl'
    with open(pkl_filename, 'wb') as pkl_file:
        pickle.dump(best_fit_genome, pkl_file)