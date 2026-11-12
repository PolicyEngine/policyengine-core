import logging

import numpy as np
import pandas as pd

from policyengine_core.country_template import Microsimulation
from policyengine_core.data import Dataset


def test__given_unknown_dataset_column__then_warns_and_still_loads(caplog):
    # Given a dataset carrying a column that matches no variable (e.g. an
    # input that was renamed or removed from the model after the dataset
    # was built).
    data = {
        "person_id__2022": [0, 1, 2],
        "person_household_id__2022": [0, 0, 1],
        "person_household_role__2022": ["parent", "child", "parent"],
        "household_weight__2022": [10.0, 10.0, 20.0],
        "salary__2022-01": [100.0, 200.0, 300.0],
        "a_removed_input__2022": [1.0, 2.0, 3.0],
    }

    # When the simulation is built from it
    with caplog.at_level(logging.WARNING):
        simulation = Microsimulation(
            dataset=Dataset.from_dataframe(pd.DataFrame(data), "2022")
        )
        salary = simulation.calculate("salary", "2022-01")

    # Then the unknown column is reported instead of vanishing silently
    warnings = [
        record.message
        for record in caplog.records
        if "a_removed_input__2022" in record.message
    ]
    assert warnings, "expected a warning naming the ignored column"
    assert "ignored" in warnings[0]

    # And the known columns still load normally
    np.testing.assert_array_equal(salary.values, np.array([100.0, 200.0, 300.0]))


def test__given_only_known_columns__then_no_unknown_column_warning(caplog):
    data = {
        "person_id__2022": [0, 1],
        "person_household_id__2022": [0, 0],
        "person_household_role__2022": ["parent", "child"],
        "household_weight__2022": [10.0, 10.0],
        "salary__2022-01": [100.0, 200.0],
    }

    with caplog.at_level(logging.WARNING):
        Microsimulation(
            dataset=Dataset.from_dataframe(pd.DataFrame(data), "2022")
        ).calculate("salary", "2022-01")

    assert not [
        record
        for record in caplog.records
        if "do not match any variable" in record.message
    ]
