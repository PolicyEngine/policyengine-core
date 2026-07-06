from typing import Iterable, Optional
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

    # The component that carries the bracket's threshold, versus the components
    # that carry its "value" (amount/rate/base). A bare ``uprating`` applies to
    # the value side; the threshold is opted in separately (issue #390).
    _threshold_key = "threshold"
    _value_keys = ("amount", "rate", "average_rate", "base")

    @staticmethod
    def allowed_unit_keys():
        return [key + "_unit" for key in ParameterScaleBracket._component_keys]

    def get_descendants(self) -> Iterable[Parameter]:
        for key in self._component_keys:
            if key in self.children:
                yield self.children[key]

    def propagate_uprating(
        self,
        uprating: str,
        threshold: bool = False,
        threshold_uprating: Optional[str] = None,
    ) -> None:
        """Route uprating metadata down onto the bracket's component leaves.

        Uprating is applied by reading each leaf ``Parameter``'s
        ``metadata["uprating"]``. Scale parameters carry their uprating intent on
        the scale, the bracket, or the individual component, so this method
        resolves all of those into per-leaf ``uprating`` entries. See issue #390.

        Resolution order for a component (highest priority first):

        1. the component leaf's own ``metadata.uprating`` (per-leaf form, e.g.
           ``brackets[i].threshold.metadata.uprating``);
        2. a bracket-level per-component override
           (``brackets[i].metadata.threshold.uprating``);
        3. a bracket-level default: ``brackets[i].metadata.uprating`` for the
           value side, and for the threshold either
           ``brackets[i].metadata.threshold_uprating`` or the bare bracket
           ``uprating`` when ``brackets[i].metadata.uprate_thresholds`` is set;
        4. the scale-level default passed in by :class:`ParameterScale`
           (``uprating`` for the value side, gated by ``uprate_thresholds`` for
           the threshold; ``threshold_uprating`` for the threshold).

        Args:
            uprating: Scale-level default uprating for the value side (may be
                ``None``).
            threshold: Whether the scale-level ``uprating`` should also apply to
                thresholds (the scale's ``uprate_thresholds`` flag).
            threshold_uprating: Scale-level uprating dedicated to thresholds (may
                be ``None``).
        """
        bracket_meta = self.metadata

        # Bracket-level defaults, falling back to the scale-level ones.
        bracket_value_uprating = bracket_meta.get("uprating", uprating)
        bracket_threshold_uprating = bracket_meta.get(
            "threshold_uprating", threshold_uprating
        )
        bracket_uprate_thresholds = bracket_meta.get("uprate_thresholds", threshold)

        for key in self._component_keys:
            if key not in self.children:
                continue
            child = self.children[key]

            # (2) bracket-level per-component override, e.g. metadata.threshold.
            component_override = None
            component_meta = bracket_meta.get(key)
            if isinstance(component_meta, dict):
                component_override = component_meta.get("uprating")

            if key == self._threshold_key:
                # Threshold: dedicated index, else bare bracket uprating when the
                # bracket/scale opts thresholds in.
                default = bracket_threshold_uprating
                if default is None and bracket_uprate_thresholds:
                    default = bracket_value_uprating
            else:
                # Value side (amount/rate/base/average_rate).
                default = bracket_value_uprating

            resolved = (
                child.metadata.get("uprating")  # (1) per-leaf wins
                or component_override  # (2)
                or default  # (3)/(4)
            )
            if resolved is not None:
                child.metadata["uprating"] = resolved
