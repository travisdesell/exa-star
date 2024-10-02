from abc import abstractmethod

import torch


class Component(torch.nn.Module):
    """
    General component class for genomes - as of writing this, basically a node or an edge. Since they share a lot of
    methods regarding their state, activation state, and parent class of `torch.nn.Module` it made sense to group them
    together. There are also several operations that function over a generic component regardless of type (node or edge)
    so this helps the type checker simplify things.

    There is also a good amount of clutter that is effectively removed from our node and edge classes, with all of these
    simple getter and setter methods.
    """

    def __init__(
        self,
        *args,
        enabled: bool = True,
        active: bool = True,
        weights_initialized: bool = False,
        **kwargs
    ) -> None:
        super().__init__(*args, *kwargs)

        # A component is active iff it is forward and backward reachable - an enabled component could be inactive
        self.active: bool = active

        # An enabled component will be included in network training iff it is active,
        # disabled components are never included
        self.enabled: bool = enabled

        # In PyTorch, there is no reliable way to know if a `Parameter` has been initialized.
        # `torch.nn.UninitializedParameter` does not seem to be a perfect fit as it does not store shape information.
        # This field should be used by weight initialization strategies to find uninitialized components, then all
        # parameters given by `self.parameters()` can be initialized properly.
        self._weights_initialized = weights_initialized

    @abstractmethod
    def __hash__(self) -> int: ...

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...

    def weights_initialized(self) -> bool:
        return self._weights_initialized

    def set_weights_initialized(self, initialized: bool) -> None:
        self._weights_initialized = initialized

    def is_active(self) -> bool:
        return self.active

    def is_inactive(self) -> bool:
        return not self.active

    def set_active(self, active: bool):
        self.active = active

    def activate(self):
        self.set_active(True)

    def deactivate(self):
        self.set_active(False)

    def is_enabled(self) -> bool:
        return self.enabled

    def is_disabled(self) -> bool:
        return not self.enabled

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def enable(self) -> None:
        self.set_enabled(True)

    def disable(self) -> None:
        self.set_enabled(False)
