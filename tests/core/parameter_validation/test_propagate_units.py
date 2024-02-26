import os

from policyengine_core.parameters import load_parameter_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def test_propagate_units():
    path = os.path.join(BASE_DIR, "parameter_for_unit_propagation.yaml")
    parameter = load_parameter_file(path)
    parameter.propagate_units()
    for i in range(4):
        assert parameter.brackets[i].threshold.metadata["unit"] == "child"
        assert parameter.brackets[i].amount.metadata["unit"] == "currency-USD"
