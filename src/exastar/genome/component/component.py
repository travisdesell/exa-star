class Component:

    def __init__(self, *args, active: bool = True, enabled: bool = True, **kwargs) -> None:
        super().__init__(*args, *kwargs)
        # A component is active iff it is forward and backward reachable - an enabled component could be inactive
        self.active: bool = active

        # An enabled component will be included in network training iff it is active,
        # disabled components are never included
        self.enabled: bool = enabled

    def is_active(self) -> bool:
        return self.active

    def set_active(self, active: bool):
        self.active = active

    def activate(self):
        self.set_active(True)

    def deactivate(self):
        self.set_active(False)

    def is_enabled(self) -> bool:
        return self.enabled

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def enable(self) -> None:
        self.set_enabled(True)

    def disable(self) -> None:
        self.set_enabled(False)
