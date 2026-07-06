from __future__ import annotations

import typing

import numpy

from policyengine_core.taxscales import AmountTaxScaleLike

if typing.TYPE_CHECKING:
    NumericalArray = typing.Union[numpy.int_, numpy.float_]


class SingleAmountTaxScale(AmountTaxScaleLike):
    def calc(
        self,
        tax_base: NumericalArray,
        right: bool = False,
    ) -> numpy.float_:
        """
        Matches the input amount to a set of brackets and returns the single
        cell value that fits within that bracket.

        The returned array preserves the dtype of the bracket amounts. In
        particular, a scale whose amounts are booleans (``amount_unit: bool``
        with ``amount: true``/``false``) returns a boolean array rather than the
        int64 ``0``/``1`` array that naive sentinel padding would produce. This
        keeps logical NOT (``~``) working on the result: ``~`` on an int array
        is a bitwise negation (``~1 == -2`` and ``~0 == -1``, both truthy),
        whereas ``~`` on a bool array is the intended element-wise logical
        negation.
        """
        guarded_thresholds = numpy.array([-numpy.inf] + self.thresholds + [numpy.inf])

        bracket_indices = numpy.digitize(
            tax_base,
            guarded_thresholds,
            right=right,
        )

        # Build the amounts array from the bracket amounts first, so numpy
        # infers their natural dtype (bool for boolean brackets, int/float
        # otherwise), then pad the two out-of-range sentinel cells with a zero
        # of that same dtype (``False`` for bool, ``0`` for numeric). Padding
        # with a Python int ``0`` up front, as the previous implementation did,
        # coerced the whole array to int64 and silently dropped a boolean dtype.
        # With no amounts, fall back to the historical int64 zero array.
        if len(self.amounts) == 0:
            guarded_amounts = numpy.array([0, 0])
        else:
            amounts = numpy.asarray(self.amounts)
            sentinel = numpy.zeros(1, dtype=amounts.dtype)
            guarded_amounts = numpy.concatenate([sentinel, amounts, sentinel])

        return guarded_amounts[bracket_indices - 1]
