from typing import Iterable
from policyengine_core.parameters import Parameter, ParameterNode


class ParameterScaleBracket(ParameterNode):
    """
    A parameter scale bracket.
    """

    _allowed_keys = set(
        ["amount", "threshold", "rate", "average_rate", "base"]
    )

    @staticmethod
    def allowed_unit_keys():
        return [key + "_unit" for key in ParameterScaleBracket._allowed_keys]

    def get_descendants(self) -> Iterable[Parameter]:
        for key in self._allowed_keys:
            if key in self.children:
                yield self.children[key]

    def propagate_uprating(
        self, uprating: str, threshold: bool = False
    ) -> None:
        for key in self._allowed_keys:
            if key in self.children:
                if key == "threshold" and not threshold:
                    continue
                self.children[key].metadata["uprating"] = (
                    uprating or self.children[key].metadata.get("uprating")
                )
