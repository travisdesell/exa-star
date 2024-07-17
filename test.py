import sys

from evolution.exagp import EXAGP

from genomes.minimal_recurrent_genome import MinimalRecurrentGenome
from genomes.trivial_recurrent_genome import TrivialRecurrentGenome

from loguru import logger

from torch import optim

from time_series.time_series import TimeSeries

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", backtrace=True, diagnose=True)

    csv_filename = (
        "/Users/travisdesell/Data/stocks/most_newest/DJI_company_2023/train/AAPL.csv"
    )

    initial_series = TimeSeries.create_from_csv(filename=csv_filename)

    print(initial_series.series_dictionary)

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

    for genome_number in range(500):
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
    print(f"{exagp.population_strategy.population[0]}")

    exagp.population_strategy.population[0].plot()
