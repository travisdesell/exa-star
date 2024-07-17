from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from config import configclass


class LogDataProvider[T]:

    def __init__(self) -> None: ...

    def prefix(self, prefix: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._modify_keys(prefix, "", data)

    def suffix(self, suffix: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._modify_keys("", suffix, data)

    def _modify_keys(
        self, prefix: str, suffix: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {f"{prefix}{k}{suffix}": v for k, v in data.items()}

    @abstractmethod
    def get_log_data(self, aggregator: T) -> Dict[str, Any]: ...


@configclass(name="base_log_data_provider", group="log_data_providers")
class LogDataProviderConfig:
    ...


class LogDataAggregator[T](LogDataProvider):

    def __init__(self, providers: Dict[str, LogDataProvider]) -> None:
        super().__init__()

        self._providers: Dict[str, LogDataProvider] = providers

    def get_log_data(self, aggregator: T) -> Dict[str, Any]:
        data: Dict[str, Any] = {}

        def append_data(new_data: Dict[str, Any]):
            intersection = new_data.keys() & data.keys()
            if intersection:
                raise ValueError(f"Log provider gave duplicate key(s): {intersection}")

            data.update(new_data)

        for _, value in self.__dict__.items():
            if not isinstance(value, LogDataProvider):
                continue

            append_data(value.get_log_data(self))

        for _, provider in self._providers.items():
            append_data(provider.get_log_data(self))

        return data


@dataclass
class LogDataAggregatorConfig:
    providers: Dict[str, LogDataProviderConfig] = field(default_factory=dict)
