from __future__ import annotations
from dataclasses import dataclass, make_dataclass
from typing import Any, Callable, Optional, Type, Union

from hydra.core.config_store import ConfigStore
from loguru import logger

# Register the config types. This is necessary so hydra can instantiate everything for us
# Only terminal classes need to be registered, as those are the classes that contain instantiation information,
# in particular the `_target_` field.
cs = ConfigStore.instance()

_config_groups = {
    "log_data_providers",
    "population",
    "genome_factory",
    "genome_factory/crossover_operators",
    "genome_factory/mutation_operators",
    "genome_factory/seed_genome_factory",
    "evolutionary_strategy",
    "evolutionary_strategy/init_task",
    "fitness",
    "dataset"
}


def fullname(ty: Any):
    """
    Given a class / type this will provide you with the fully qualified package path + name.
    e.g. foo.bar.MyClass

    Taken from:
    https://stackoverflow.com/questions/2020014/get-fully-qualified-class-name-of-an-object-in-python
    """
    module = ty.__module__
    if module == 'builtins':
        return ty.__qualname__
    return module + '.' + ty.__qualname__


def configclass(
        name: Optional[str] = None,
        group: Optional[str] = None,
        target: Optional[Union[Type, Callable]] = None,
        **dataclass_args):
    def mk(config_class):
        if group and group not in _config_groups:
            logger.warning(f"The supplied group for class {config_class} is not in the set of _config_groups")
            logger.warning(f"group {group} / name {name}")

        data_class = dataclass(**dataclass_args)(config_class)

        if target:
            @dataclass
            class _Target(data_class):
                _target_: str = fullname(target)
            _Target.__name__ = data_class.__name__
            data_class = _Target

        if name:
            cs.store(group=group, name=name, node=data_class)

        return data_class

    return mk
