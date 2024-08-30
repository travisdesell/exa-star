# Some predicates
def is_none(x):
    return x is None


def is_not_none(x):
    return x is not None


def is_instance(ty):
    return lambda x: isinstance(x, ty)


def is_not_instance(ty):
    return lambda x: not isinstance(x, ty)
