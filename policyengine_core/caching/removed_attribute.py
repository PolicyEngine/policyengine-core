"""Explicit rejection of removed cache attributes, including legacy writes."""


class RemovedCacheAttribute:
    """A data descriptor that prevents obsolete names becoming shadow state."""

    def __init__(self, replacement: str) -> None:
        self.replacement = replacement
        self.name = "removed cache attribute"

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        raise AttributeError(f"{self.name} was removed; use {self.replacement}")

    def __set__(self, instance, value) -> None:
        raise AttributeError(f"{self.name} was removed; use {self.replacement}")
