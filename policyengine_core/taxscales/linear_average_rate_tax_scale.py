from __future__ import annotations

import logging
import typing

import numpy

from policyengine_core.taxscales.marginal_rate_tax_scale import (
    MarginalRateTaxScale,
)
from policyengine_core.taxscales.rate_tax_scale_like import RateTaxScaleLike

log = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    NumericalArray = typing.Union[numpy.int_, numpy.float_]


class LinearAverageRateTaxScale(RateTaxScaleLike):
    def calc(
        self,
        tax_base: NumericalArray,
        right: bool = False,
    ) -> numpy.float_:
        """
        Compute the tax amount for the given tax bases by applying a taxscale.

        Each bracket's rate is the average rate at its threshold. Between two
        thresholds the average rate moves linearly from one rate to the next.
        From the last threshold up it stays at the last rate. Below the first
        threshold the tax is 0, except that a scale with a single bracket
        applies its rate to every tax base.

        :param ndarray tax_base: Array of the tax bases.

        :returns: Float array with tax amount for the given tax bases.

        For instance:

        >>> tax_scale = LinearAverageRateTaxScale()
        >>> tax_scale.add_bracket(0, 0)
        >>> tax_scale.add_bracket(100, 0.1)
        >>> tax_base = array([50, 100, 150])
        >>> tax_scale.calc(tax_base)
        [2.5, 10.0, 15.0]
        """
        if len(self.rates) == 1:
            return tax_base * self.rates[0]

        tiled_base = numpy.tile(tax_base, (len(self.thresholds) - 1, 1)).T
        tiled_thresholds = numpy.tile(self.thresholds, (len(tax_base), 1))

        bracket_dummy = (tiled_base >= tiled_thresholds[:, :-1]) * (
            +tiled_base < tiled_thresholds[:, 1:]
        )

        rates_array = numpy.array(self.rates)
        thresholds_array = numpy.array(self.thresholds)

        rate_slope = (rates_array[1:] - rates_array[:-1]) / (
            +thresholds_array[1:] - thresholds_array[:-1]
        )

        average_rate_slope = numpy.dot(bracket_dummy, rate_slope.T)

        bracket_average_start_rate = numpy.dot(bracket_dummy, rates_array[:-1])
        bracket_threshold = numpy.dot(bracket_dummy, thresholds_array[:-1])

        log.info(f"bracket_average_start_rate :  {bracket_average_start_rate}")
        log.info(f"average_rate_slope:  {average_rate_slope}")

        # The brackets above end at the last threshold, so bases from there up
        # fall in none of them. With no next rate to move toward, their average
        # rate stays at the last rate.
        above_last_threshold = numpy.asarray(tax_base) >= thresholds_array[-1]

        return tax_base * (
            +bracket_average_start_rate
            + (tax_base - bracket_threshold) * average_rate_slope
            + above_last_threshold * rates_array[-1]
        )

    def to_marginal(self) -> MarginalRateTaxScale:
        marginal_tax_scale = MarginalRateTaxScale(
            name=self.name,
            option=self.option,
            unit=self.unit,
        )

        previous_i = 0
        previous_threshold = 0

        for threshold, rate in zip(self.thresholds[1:], self.rates[1:]):
            if threshold != float("Inf"):
                i = rate * threshold
                marginal_tax_scale.add_bracket(
                    previous_threshold,
                    (i - previous_i) / (threshold - previous_threshold),
                )
                previous_i = i
                previous_threshold = threshold

        marginal_tax_scale.add_bracket(previous_threshold, rate)

        return marginal_tax_scale
