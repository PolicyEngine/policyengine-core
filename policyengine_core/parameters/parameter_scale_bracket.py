from typing import Iterable
from policyengine_core.parameters import Parameter, ParameterNode


class ParameterScaleBracket(ParameterNode):
    """
    A parameter scale bracket.
    """

    _allowed_keys = set(
        ["amount", "threshold", "rate", "average_rate", "base"]
    )

    def get_descendants(self) -> Iterable[Parameter]:
        for key in self._allowed_keys:
            if key in self.children:
                yield self.children[key]
