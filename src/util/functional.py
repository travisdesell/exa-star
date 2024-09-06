# Some predicates
from typing import Set


def is_none(x):
    return x is None


def is_not_none(x):
    return x is not None


def is_instance(ty):
    """
    Returns a lambda which returns true if the supplied value `x` is a descendent of or is exactly the type `ty`
    """
    return lambda x: isinstance(x, ty)


def is_not_instance(ty):
    return lambda x: not isinstance(x, ty)


def is_type(ty):
    """
    Returns a lambda which will return true if the supplied value has the exact type of `ty`.
    If you want to consider all classes that derive `ty` as well use `is_instance`
    """
    return lambda x: type(x) is ty


def is_not_type(ty):
    return lambda x: type(x) is not ty


def is_any_instance(types: Set[type]):
    return lambda x: any(isinstance(x, ty) for ty in types)


def is_not_any_instance(types: Set[type]):
    return lambda x: not any(isinstance(x, ty) for ty in types)


def is_any_type(types: Set[type]):
    return lambda x: type(x) in types


def is_not_any_type(types: Set[type]):
    return lambda x: type(x) not in types
