from __future__ import annotations
from dataclasses import field
from typing import List, Tuple

from dataset import Dataset, DatasetConfig
from config import configclass

from loguru import logger
import numpy as np
import pandas as pd
import torch


class TimeSeries(Dataset):
    def __init__(
        self,
        series_dictionary: dict[str, torch.Tensor],
        input_series_names: List[str],
        output_series_names: List[str]
    ) -> None:
        """
        Initializes a time series object for time series tasks and verifies that
        the time series are structured correctly.  Either a filename or a pre-processed
        dictionary needs to be provided.

        Args:
            series_dictionary: is a dictionary of column names to tensors of values (1 per time step).
        """
        self.series_dictionary = series_dictionary

        self.input_series_names: List[str] = input_series_names
        if not input_series_names:
            logger.warning("input_series_names is empty!")

        self.output_series_names: List[str] = output_series_names
        if not output_series_names:
            logger.warning("output_series_names is empty!")

        # validate that the length of each series is the same and save that length
        self.series_length: int = -1

        for series_name, series in self.series_dictionary.items():
            shape = series.shape

            # each series should be a 1D tensor
            assert len(shape) == 1

            if self.series_length == -1:
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
    def create_truncated_from_csv(
        filenames: List[str],
        input_series: List[str],
        output_series: List[str],
        length: int
    ) -> TimeSeries:
        series = TimeSeries.create_from_csv(filenames, input_series, output_series)
        return series.slice(0, length)

    @staticmethod
    def create_from_csv(filenames: List[str], input_series: List[str], output_series: List[str]) -> TimeSeries:
        """
        Initializes a TimeSeries object from a CSV file.

        Args:
            filename: is the CSV filename.
        """
        filename = filenames[0]
        csv_dict = pd.read_csv(filename, encoding="UTF-8")

        # convert the pandas dataframe to a dict of pytorch tensors

        series_dictionary = {}
        for series_name, values in csv_dict.items():
            if values.dtype.type in {
                np.float64,
                np.float32,
                np.float16,
                np.complex64,
                np.complex128,
                np.int64,
                np.int32,
                np.int16,
                np.int8,
                np.uint8,
                np.bool,
            }:
                series_dictionary[series_name] = torch.tensor(values.to_numpy())
            else:
                logger.warning(
                    f"not including series '{series_name}' in TimeSeries object because the type "
                    f"'{values.dtype}' cannot be converted to a pytorch tensor."
                )

        if not input_series:
            input_series = list(series_dictionary.keys())

        if not output_series:
            output_series = list(series_dictionary.keys())

        return TimeSeries(series_dictionary, input_series, output_series)

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

        return TimeSeries(input_series, self.input_series_names, self.output_series_names)

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

        return TimeSeries(output_series, self.input_series_names, self.output_series_names)

    def slice(self, start_row: int, end_row: int) -> TimeSeries:
        """
        Copies a time slice of this time series and returns it as a new time series.

        Args:
            start_row: the first row of the time slice
            end_row: the last row of the time slice (non-inclusive)

        Returns:
            A time series where each parameter only has the values between start_row and end_row.
        """
        slice_series_dictionary = {}

        for series_name, values in self.series_dictionary.items():
            slice_series_dictionary[series_name] = values[start_row:end_row].clone()

        return TimeSeries(slice_series_dictionary, self.input_series_names, self.output_series_names)

    def get_series_names(self) -> List[str]:
        return list(self.series_dictionary.keys())


@configclass(name="base_time_series_dataset", group="dataset", target=TimeSeries.create_from_csv)
class TimeSeriesConfig(DatasetConfig):
    filenames: Tuple[str, ...]
    input_series: List[str] = field(default_factory=lambda: [])
    output_series: List[str] = field(default_factory=lambda: [])


@configclass(name="base_aapl_time_series_dataset", group="dataset", target=TimeSeries.create_from_csv)
class AAPLTimeSeriesConfig(TimeSeriesConfig):
    filenames: Tuple[str, ...] = (
        "~/Downloads/aapl.csv",
    )


@configclass(name="base_aapl_time_series_dataset", group="dataset", target=TimeSeries.create_truncated_from_csv)
class TestDataset(TimeSeriesConfig):
    filenames: Tuple[str, ...] = (
        "~/Downloads/aapl.csv",
    )
    length: int = 128
