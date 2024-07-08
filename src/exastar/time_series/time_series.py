from __future__ import annotations

import pandas as pd
import torch

from loguru import logger


class TimeSeries:
    def __init__(self, series_dictionary: dict[str, torch.Tensor]):
        """
        Initializes a time series object for time series tasks and verifies that
        the time series are structured correctly.  Either a filename or a pre-processed
        dictionary needs to be provided.

        Args:
            series_dictionary: is a dictionary of column names to tensors of values (1 per
                time step).
        """
        self.series_dictionary = series_dictionary

        # validate that the length of each series is the same and save that length
        self.series_length = None
        for series_name, series in self.series_dictionary.items():
            shape = series.shape
            print(f"'{series_name}' shape {shape}")

            # each series should be a 1D tensor
            assert len(shape) == 1

            if self.series_length is None:
                self.series_length = shape[0]
            else:
                if self.series_length != shape[0]:
                    logger.error(
                        "TimeSeries tensors should all be the same length. Tensor dictionary shapes were:"
                    )
                    for series_name, series in self.series_dictionary.items():
                        logger.error(f"\t'{series_name}': {series.shape}")
                    exit(1)

    @staticmethod
    def create_from_csv(filename: str) -> TimeSeries:
        """
        Initializes a TimeSeries object from a CSV file.

        Args:
            filename: is the CSV filename.
        """
        csv_dict = pd.read_csv(filename, encoding="UTF-8")

        # convert the pandas dataframe to a dict of pytorch tensors

        series_dictionary = {}
        for series_name, values in csv_dict.items():
            logger.info(f"{series_name} type: {values.dtype}")
            if values.dtype in [
                "float64",
                "float32",
                "float16",
                "complex64",
                "complex128",
                "int64",
                "int32",
                "int16",
                "int8",
                "uint8",
                "bool",
            ]:
                series_dictionary[series_name] = torch.tensor(values.to_numpy())
            else:
                logger.warning(
                    f"not including series '{series_name}' in TimeSeries object because the type "
                    f"'{values.dtype}' cannot be converted to a pytorch tensor."
                )

        return TimeSeries(series_dictionary=series_dictionary)

    def get_inputs(self, input_series_names: list[str], offset: int) -> TimeSeries:
        """
        Return:
            A TimeSeries object which is a subset of this time series with
            the given series names, with the length reduced by the offset
            so it can be paired with a different output time series as
            expected values.
        """
        input_series = {}

        for series_name in input_series_names:
            input_series[series_name] = self.series_dictionary[series_name][offset:]

        return TimeSeries(series_dictionary=input_series)

    def get_outputs(self, output_series_names: list[str], offset: int) -> TimeSeries:
        """
        Return:
            A TimeSeries object which is a subset of this time series with
            the given series names, with the length shifted up by the offset
            so it can be paired with a different input time series as
            expected values.
        """
        output_series = {}

        for series_name in output_series_names:
            output_series[series_name] = self.series_dictionary[series_name][:-offset]

        return TimeSeries(series_dictionary=output_series)

    def slice(self, start_row: int, end_row: int) -> TimeSeries:
        """Copies a time slice of this time series and returns it as a new time series.

        Args:
            start_row: the first row of the time slice
            end_row: the last row of the time slice (non-inclusive)

        Returns:
            A time series where each parameter only has the values between start_row and end_row.
        """
        slice_series_dictionary = {}

        for series_name, values in self.series_dictionary.items():
            slice_series_dictionary[series_name] = values[start_row:end_row].clone()

        return TimeSeries(slice_series_dictionary)
