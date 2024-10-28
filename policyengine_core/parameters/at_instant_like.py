import abc
from typing import Any

from policyengine_core import periods
from policyengine_core.periods import Instant


class AtInstantLike(abc.ABC):
    """
    Base class for various types of parameters implementing the at instant protocol.
    """

    def __call__(self, instant: Instant) -> Any:
        return self.get_at_instant(instant)

    def get_at_instant(self, instant: Instant) -> Any:
        instant = str(periods.instant(instant))
        return self._get_at_instant(instant)

    @abc.abstractmethod
    def _get_at_instant(self, instant): ...

    def get_attr_dict(self) -> dict:
        attr_dict = {}
        attr_list = [
            "name",
            "description",
            "documentation",
            "file_path",
            "metadata",
            "trace",
            "tracer",
            "branch_name",
            "modified",
            "values_list",
        ]
        for attr in attr_list:
            if hasattr(self, attr):
                attr_dict[attr] = getattr(self, attr)
        if hasattr(self, "children"):
            for child_name, child in self.children.items():
                attr_dict[child_name] = child.get_attr_dict()
        if hasattr(self, "values_list"):
            value_dict = {}
            attr_dict["values_list"] = value_dict
            for value_at_instant in self.values_list:
                value_dict[value_at_instant.instant_str] = (
                    value_at_instant.value
                )
        return attr_dict
