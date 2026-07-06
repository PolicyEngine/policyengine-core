from typing import Iterable
from policyengine_core.parameters import Parameter, ParameterNode
from policyengine_core.parameters.config import COMMON_KEYS


class ParameterScaleBracket(ParameterNode):
    """
    A parameter scale bracket.
    """

    # The tax-scale components a bracket can hold as children.
    _component_keys = ("amount", "threshold", "rate", "average_rate", "base")

    # Keys accepted on a bracket node. Beyond the components, brackets may carry
    # the common ``description``/``metadata``/``documentation`` keys (metadata is
    # used to route uprating to the components; see issue #390). Any other key is
    # flagged by the loader (issue #505).
    _allowed_keys = frozenset(_component_keys).union(COMMON_KEYS)

    @staticmethod
    def allowed_unit_keys():
        return [key + "_unit" for key in ParameterScaleBracket._component_keys]

    def get_descendants(self) -> Iterable[Parameter]:
        for key in self._component_keys:
            if key in self.children:
                yield self.children[key]

    def propagate_uprating(self, uprating: str, threshold: bool = False) -> None:
        for key in self._component_keys:
            if key in self.children:
                if key == "threshold" and not threshold:
                    continue
                self.children[key].metadata["uprating"] = uprating or self.children[
                    key
                ].metadata.get("uprating")
