import pytest

from policyengine_core.commons.formulas import random
from policyengine_core.model_api import random as model_api_random


@pytest.mark.parametrize("random_function", [random, model_api_random])
def test_random_raises_for_formula_time_randomness(random_function):
    with pytest.raises(RuntimeError, match="Formula-time randomness is not allowed"):
        random_function(None)
