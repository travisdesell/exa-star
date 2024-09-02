import torch


class Component(torch.nn.Module):

    def __init__(
        self,
        *args,
        active: bool = True,
        enabled: bool = True,
        weights_initialized: bool = False,
        **kwargs
    ) -> None:
        super().__init__(*args, *kwargs)

        # A component is active iff it is forward and backward reachable - an enabled component could be inactive
        self.active: bool = active

        # An enabled component will be included in network training iff it is active,
        # disabled components are never included
        self.enabled: bool = enabled

        self._weights_initialized = weights_initialized

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
