from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from config import configclass


class LogDataProvider[T]:
    """
    A log data provider is exactly what it sounds like: an object that provides log data.
    This interface should be used like a trait on objects you want to collect log data for.
    See LogDataAggregator to see how this is done.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def prefix(self, prefix: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._modify_keys(prefix, "", data)

    def suffix(self, suffix: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._modify_keys("", suffix, data)

    def _modify_keys(
        self, prefix: str, suffix: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {f"{prefix}{k}{suffix}": v for k, v in data.items()}

    @abstractmethod
    def get_log_data(self, aggregator: T) -> Dict[str, Any]:
        """
        Args:
            aggregator: A reference to the aggregator this LogDataProvider is associated with (see below).
              This value may be necessary to obtain the data you want to log.

        Returns:
            A dictionary of log data. The keys in this dictionary should be unique.
        """
        ...


@configclass(name="base_log_data_provider", group="log_data_providers")
class LogDataProviderConfig:
    ...


class LogDataAggregator[T](LogDataProvider[T]):
    """
    A log data aggregator will contain any number of `LogDataProvider` objects. Log data will be recorded from each of
    these objects every time `get_log_data` is called. Note that this is done automatically by iterating through the
    attributes of the `LogDataAggregator` object.

    There is an additional way to grab log data, and that is by explicitly specifying log data providers. Each one of
    these will of course have `get_log_data` called, and `self`.
    """

    def __init__(self, providers: Dict[str, LogDataProvider]) -> None:
        super().__init__()

        self._providers: Dict[str, LogDataProvider] = providers

    def get_log_data(self, aggregator: T) -> Dict[str, Any]:
        """
        Args:
            aggregator: Unused, same as `LogDataProvider`

        Returns:
            A dictionary of log data. The keys in this dictionary should be unique. This dictionary is created by
            merging the log data from all of the `LogDataProvider` attributes this object contains, as well as the log
            data from the `LogDataPRovider` objects in `self._providers`.
        """
        data: Dict[str, Any] = {}

        def append_data(new_data: Dict[str, Any]):
            intersection = new_data.keys() & data.keys()
            if intersection:
                raise ValueError(f"Log provider gave duplicate key(s): {intersection}")

            data.update(new_data)

        # Crawl through all of the attributes of `self` and look for `LogDataProvider` objects.
        for _, value in self.__dict__.items():
            # Gather log data if `value` is a `LogDataProvider`.
            if isinstance(value, LogDataProvider):
                append_data(value.get_log_data(self))

        # Gather log data from each of the providers in self._providers.
        for _, provider in self._providers.items():
            append_data(provider.get_log_data(self))

        return data


@dataclass
class LogDataAggregatorConfig:
    providers: Dict[str, LogDataProviderConfig] = field(default_factory=dict)
