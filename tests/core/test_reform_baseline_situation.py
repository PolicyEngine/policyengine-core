"""A reform simulation's baseline holds what its own system makes of the situation.

``Simulation(situation=..., reform=...)`` builds the situation under the
reform's system, then makes its baseline a branch of that simulation under the
baseline system. A stored input is a whole array: the situation builder gives
the people a situation gives no value for a variable it gives others (or sets
along axes) that variable's default, from the system it builds under. The
branch kept those arrays, so the baseline read the reform's default for those
people (PolicyEngine/policyengine-core#612). It also had no input for a
variable the reform neutralizes, whose input the reform's system ignores. The
baseline's inputs are now built from the situation under the baseline system,
leaving out the inputs that system cannot take.

The property over any reform and situation is in
``test_reform_baseline_situation_property.py``.
"""

from __future__ import annotations

import copy
import datetime

import numpy as np
import pytest

from policyengine_core.country_template import Simulation
from policyengine_core.enums import Enum
from policyengine_core.errors import SituationParsingError, VariableNotFoundError
from policyengine_core.periods import ETERNITY, period
from policyengine_core.reforms import Reform
from policyengine_core.simulations import SimulationBuilder
from policyengine_core.variables import Variable
from tests.core.test_reform_baseline_system import (
    REFORM_DEFAULT_SALARY,
    add_doubled_salary,
    change_salary_default,
    move_salary_to_households,
    neutralize_income_tax,
)

PERIOD = "2022-01"
RATE = 0.15
SYSTEM = Simulation.default_tax_benefit_system_instance
OUTPUTS = [
    "salary",
    "income_tax",
    "disposable_income",
    "rent",
    "housing_occupancy_status",
    "total_taxes",
    "household_income",
]
PARTIAL = {
    "persons": {"a": {"salary": {PERIOD: 3000.0}}, "b": {}},
    "households": {"h": {"parents": ["a", "b"]}},
}
REFORM_RENT = 600.0
REFORM_BIRTH = datetime.date(1990, 6, 1)


class rent(Variable):
    default_value = REFORM_RENT


class birth(Variable):
    default_value = REFORM_BIRTH


class change_household_defaults(Reform):
    def apply(self):
        self.update_variable(rent)
        # Each system loads its own enum classes, so the default is a member
        # of this system's.
        status = self.get_variable("housing_occupancy_status").possible_values
        self.update_variable(
            type(
                "housing_occupancy_status", (Variable,), {"default_value": status.owner}
            )
        )


class change_birth_default(Reform):
    def apply(self):
        self.update_variable(birth)


class neutralize_salary(Reform):
    def apply(self):
        self.neutralize_variable("salary")


class ExtendedStatus(Enum):
    owner = "Owner"
    tenant = "Tenant"
    free_lodger = "Free lodger"
    homeless = "Homeless"
    student = "Student accommodation"


class extend_housing_occupancy_status(Reform):
    def apply(self):
        # ``possible_values`` and ``default_value`` are read from the class
        # itself, not from its bases.
        self.update_variable(
            type(
                "housing_occupancy_status",
                (Variable,),
                {
                    "possible_values": ExtendedStatus,
                    "default_value": ExtendedStatus.tenant,
                },
            )
        )


class eternal_salary(Variable):
    definition_period = ETERNITY


class make_salary_eternal(Reform):
    def apply(self):
        self.update_variable(type("salary", (eternal_salary,), {}))


def _built(situation, system=SYSTEM):
    """What the situation builder alone makes of ``situation`` under ``system``."""
    builder = SimulationBuilder()
    builder.default_period = PERIOD
    return builder.build_from_dict(system, copy.deepcopy(situation))


def _values(simulation, variable, period=PERIOD):
    return np.asarray(simulation.calculate(variable, period))


def assert_same(simulation, expected, outputs=OUTPUTS, periods=(PERIOD,)):
    for variable in outputs:
        for period in periods:
            np.testing.assert_array_equal(
                _values(simulation, variable, period),
                _values(expected, variable, period),
                err_msg=f"{variable} at {period}",
            )


def test_people_given_no_salary_take_the_baselines_default():
    reform = Simulation(situation=PARTIAL, reform=change_salary_default)

    np.testing.assert_array_equal(_values(reform.baseline, "salary"), [3000, 0])
    np.testing.assert_allclose(
        _values(reform.baseline, "income_tax"), [3000 * RATE, 0], rtol=1e-6
    )
    # The reform simulation keeps the reform's default.
    np.testing.assert_array_equal(
        _values(reform, "salary"), [3000, REFORM_DEFAULT_SALARY]
    )
    np.testing.assert_allclose(
        _values(reform, "income_tax"),
        [3000 * RATE, REFORM_DEFAULT_SALARY * RATE],
        rtol=1e-6,
    )
    assert_same(reform.baseline, _built(PARTIAL))
    reform_system = Simulation.default_tax_benefit_system(reform=change_salary_default)
    assert_same(reform, _built(PARTIAL, reform_system))


def test_people_off_an_axis_take_the_baselines_default():
    situation = {
        "persons": {"a": {}, "b": {}},
        "households": {"h": {"parents": ["a", "b"]}},
        "axes": [
            [{"name": "salary", "count": 2, "min": 0, "max": 5000, "period": PERIOD}]
        ],
    }
    reform = Simulation(situation=situation, reform=change_salary_default)

    np.testing.assert_array_equal(_values(reform.baseline, "salary"), [0, 0, 5000, 0])
    np.testing.assert_array_equal(
        _values(reform, "salary"),
        [0, REFORM_DEFAULT_SALARY, 5000, REFORM_DEFAULT_SALARY],
    )
    assert reform.baseline.has_axes
    assert_same(reform.baseline, _built(situation))


def test_a_default_given_for_one_month_changes_how_a_year_is_divided():
    """Why the baseline is built again rather than given other defaults.

    ``salary`` divides a yearly input between the months that have no input.
    ``b`` is given January's salary and ``a`` the year's, so January's input
    gives ``a`` the default and the year's gives ``b`` the default. What is
    left of each yearly value for February to December depends on the
    default given for January.
    """
    situation = {
        "persons": {
            "a": {"salary": {"2022": 12_000.0}},
            "b": {"salary": {PERIOD: 3000.0}},
        },
        "households": {"h": {"parents": ["a", "b"]}},
    }
    reform = Simulation(situation=situation, reform=change_salary_default)
    default = REFORM_DEFAULT_SALARY

    np.testing.assert_allclose(
        _values(reform.baseline, "salary", "2022-02"),
        [12_000 / 11, (0 - 3000) / 11],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        _values(reform, "salary", "2022-02"),
        [(12_000 - default) / 11, (default - 3000) / 11],
        rtol=1e-6,
    )
    assert_same(reform.baseline, _built(situation), periods=(PERIOD, "2022-02"))


def test_households_given_nothing_take_the_baselines_defaults():
    situation = {
        "persons": {"a": {}, "b": {}},
        "households": {
            "h": {
                "parents": ["a"],
                "rent": {PERIOD: 900.0},
                "housing_occupancy_status": {PERIOD: "free_lodger"},
            },
            "g": {"parents": ["b"]},
        },
    }
    reform = Simulation(situation=situation, reform=change_household_defaults)

    np.testing.assert_array_equal(_values(reform.baseline, "rent"), [900, 0])
    assert list(
        reform.baseline.calculate("housing_occupancy_status", PERIOD).decode_to_str()
    ) == ["free_lodger", "tenant"]
    np.testing.assert_array_equal(_values(reform, "rent"), [900, REFORM_RENT])
    assert list(
        reform.calculate("housing_occupancy_status", PERIOD).decode_to_str()
    ) == ["free_lodger", "owner"]
    assert_same(reform.baseline, _built(situation))


def test_people_given_no_birth_date_take_the_baselines_default():
    situation = {
        "persons": {"a": {"birth": {"ETERNITY": "1980-01-01"}}, "b": {}},
        "households": {"h": {"parents": ["a", "b"]}},
    }
    reform = Simulation(situation=situation, reform=change_birth_default)

    np.testing.assert_array_equal(
        _values(reform.baseline, "birth"),
        np.array(["1980-01-01", "1970-01-01"], dtype="datetime64[D]"),
    )
    np.testing.assert_array_equal(
        _values(reform, "birth"),
        np.array(["1980-01-01", REFORM_BIRTH], dtype="datetime64[D]"),
    )
    assert_same(reform.baseline, _built(situation), outputs=["birth", "age"])


@pytest.mark.filterwarnings("ignore:You cannot set a value")
@pytest.mark.parametrize(
    "situation",
    [
        pytest.param(
            {
                "persons": {
                    "a": {"salary": {PERIOD: 3000.0}, "income_tax": {PERIOD: 7.0}}
                },
                "households": {"h": {"parents": ["a"]}},
            },
            id="entities",
        ),
        # Without entities, the builder sets each variable through
        # ``Simulation.set_input`` for one person.
        pytest.param(
            {"salary": {PERIOD: 3000.0}, "income_tax": {PERIOD: 7.0}},
            id="variables",
        ),
    ],
)
def test_an_input_the_reform_neutralizes_is_the_baselines(situation):
    reform = Simulation(situation=situation, reform=neutralize_income_tax)

    np.testing.assert_array_equal(_values(reform.baseline, "income_tax"), [7])
    np.testing.assert_array_equal(_values(reform, "income_tax"), [0])
    assert "income_tax" in reform.baseline.input_variables
    assert_same(reform.baseline, _built(situation), outputs=["salary", "income_tax"])


def test_a_situation_without_entities_is_stored_as_in_a_simulation_built_from_it():
    """The baseline stores the inputs under ``"default"``, not its own name."""
    situation = {"salary": {PERIOD: [3000.0, 100.0]}}
    baseline = Simulation(situation=situation, reform=change_salary_default).baseline
    expected = Simulation(situation=situation)

    assert baseline.branch_name == "baseline"
    assert baseline._user_input_keys == expected._user_input_keys
    np.testing.assert_array_equal(
        baseline.get_holder("salary").get_array(period(PERIOD)), [3000, 100]
    )


@pytest.mark.filterwarnings("ignore:You cannot set a value")
def test_a_neutralized_input_variable_is_the_baselines():
    reform = Simulation(situation=PARTIAL, reform=neutralize_salary)

    np.testing.assert_array_equal(_values(reform.baseline, "salary"), [3000, 0])
    np.testing.assert_array_equal(_values(reform, "salary"), [0, 0])
    assert_same(reform.baseline, _built(PARTIAL))


def _without(situation, path):
    """``situation`` without the entry at ``path``."""
    stripped = copy.deepcopy(situation)
    entry = stripped
    for key in path[:-1]:
        entry = entry[key]
    del entry[path[-1]]
    return stripped


@pytest.mark.parametrize(
    "reform, situation, rejected",
    [
        pytest.param(
            add_doubled_salary,
            {
                "persons": {
                    "a": {"salary": {PERIOD: 3000.0}, "doubled_salary": {PERIOD: 5.0}},
                    "b": {"salary": {PERIOD: 100.0}},
                },
                "households": {"h": {"parents": ["a", "b"]}},
            },
            ["persons", "a", "doubled_salary"],
            id="variable-only-the-reform-has",
        ),
        pytest.param(
            move_salary_to_households,
            {
                "persons": {"a": {}, "b": {}},
                "households": {"h": {"parents": ["a", "b"], "salary": {PERIOD: 100.0}}},
            },
            ["households", "h", "salary"],
            id="variable-the-reform-moves-to-another-entity",
        ),
        pytest.param(
            extend_housing_occupancy_status,
            {
                "persons": {"a": {}, "b": {}},
                "households": {
                    "h": {
                        "parents": ["a"],
                        "housing_occupancy_status": {PERIOD: "student"},
                    },
                    "g": {
                        "parents": ["b"],
                        "housing_occupancy_status": {PERIOD: "owner"},
                    },
                },
            },
            ["households", "h", "housing_occupancy_status"],
            id="value-only-the-reform-has",
        ),
        pytest.param(
            make_salary_eternal,
            {
                "persons": {"a": {"salary": {"ETERNITY": 3000.0}}, "b": {}},
                "households": {"h": {"parents": ["a", "b"]}},
            },
            ["persons", "a", "salary"],
            id="period-only-the-reform-takes",
        ),
    ],
)
def test_inputs_the_baseline_system_cannot_take_are_left_out(
    reform, situation, rejected
):
    # Under the baseline system, the situation does not build at all.
    with pytest.raises(SituationParsingError):
        _built(situation)

    simulation = Simulation(situation=situation, reform=reform)

    assert_same(simulation.baseline, _built(_without(situation, rejected)))
    assert sorted(simulation.baseline.input_variables) == sorted(
        _built(_without(situation, rejected)).input_variables
    )


def test_an_axis_over_a_variable_only_the_reform_has_still_shapes_the_baseline():
    base = {
        "persons": {"a": {"salary": {PERIOD: 3000.0}}, "b": {}},
        "households": {"h": {"parents": ["a", "b"]}},
    }
    situation = {
        **base,
        "axes": [
            [
                {
                    "name": "doubled_salary",
                    "count": 3,
                    "min": 0,
                    "max": 10,
                    "period": PERIOD,
                }
            ],
            [
                {
                    "name": "salary",
                    "count": 2,
                    "min": 0,
                    "max": 500,
                    "period": PERIOD,
                    "index": 1,
                }
            ],
        ],
    }
    reform = Simulation(situation=situation, reform=add_doubled_salary)
    baseline = reform.baseline

    assert baseline.persons.count == reform.persons.count == 12
    assert baseline.household.count == reform.household.count == 6
    np.testing.assert_array_equal(baseline.persons.ids, reform.persons.ids)
    np.testing.assert_array_equal(
        baseline.household.members_entity_id, reform.household.members_entity_id
    )
    # ``salary``'s axis is set as it is in the reform simulation, whose
    # ``salary`` is the baseline's.
    np.testing.assert_array_equal(
        _values(baseline, "salary"), _values(reform, "salary")
    )
    assert "doubled_salary" not in baseline.input_variables
    assert all(key[0] != "doubled_salary" for key in baseline._user_input_keys)


def test_the_baseline_records_and_exports_the_inputs_its_system_builds():
    # The axis sets ``a``'s salary in each copy; ``b``, given none, takes the
    # default in each.
    situation = {
        **PARTIAL,
        "axes": [
            [
                {
                    "name": "salary",
                    "count": 2,
                    "min": 0,
                    "max": 5000,
                    "period": PERIOD,
                }
            ]
        ],
    }
    baseline = Simulation(situation=situation, reform=change_salary_default).baseline
    expected = _built(situation)

    assert baseline._user_input_keys == expected._user_input_keys
    assert sorted(baseline.input_variables) == sorted(expected.input_variables)
    assert baseline.to_input_dict() == expected.to_input_dict()
    assert baseline.to_input_dataframe().equals(expected.to_input_dataframe())
    assert baseline.has_axes == Simulation(situation=situation).has_axes


@pytest.mark.parametrize("trace", [False, True])
def test_the_rebuilt_baseline_is_still_the_reform_simulations_baseline_branch(trace):
    reform = Simulation(situation=PARTIAL, reform=change_salary_default, trace=trace)
    baseline = reform.baseline

    assert reform.branches["baseline"] is baseline
    assert reform.get_branch("baseline") is baseline
    assert baseline.parent_branch is reform
    assert baseline.branch_name == "baseline"
    assert baseline.baseline is None
    assert baseline.tracer is reform.tracer
    assert baseline.tax_benefit_system is SYSTEM
    entities = {
        entity.key: entity for entity in [SYSTEM.person_entity, *SYSTEM.group_entities]
    }
    for key, population in baseline.populations.items():
        assert population.simulation is baseline
        assert population.entity is entities[key]
        assert getattr(baseline, key) is population
    # A clone's copy of the baseline is a copy of the rebuilt one.
    np.testing.assert_array_equal(_values(reform.clone().baseline, "salary"), [3000, 0])


def test_the_baseline_keeps_its_inputs_through_the_reform_simulations_reform():
    """policyengine-us applies structural reforms after ``__init__``."""
    reform = Simulation(situation=PARTIAL, reform=change_salary_default)
    reform.apply_reform(neutralize_income_tax)

    np.testing.assert_array_equal(_values(reform.baseline, "salary"), [3000, 0])
    np.testing.assert_allclose(
        _values(reform.baseline, "income_tax"), [3000 * RATE, 0], rtol=1e-6
    )
    np.testing.assert_array_equal(_values(reform, "income_tax"), [0, 0])


def test_inputs_set_on_the_reform_simulation_afterwards_stay_out_of_the_baseline():
    reform = Simulation(situation=PARTIAL, reform=change_salary_default)
    reform.set_input("salary", PERIOD, [10.0, 20.0])

    np.testing.assert_array_equal(_values(reform, "salary"), [10, 20])
    np.testing.assert_array_equal(_values(reform.baseline, "salary"), [3000, 0])


def _strict_and_skipping():
    strict = SimulationBuilder()
    strict.default_period = PERIOD
    skipping = SimulationBuilder()
    skipping.default_period = PERIOD
    skipping.skip_rejected_inputs = True
    return strict, skipping


@pytest.mark.parametrize(
    "situation, rejected",
    [
        pytest.param(
            {
                "persons": {
                    "a": {"salary": {PERIOD: 1.0}, "no_such_variable": {PERIOD: 1}}
                },
                "households": {"h": {"parents": ["a"]}},
            },
            ["persons", "a", "no_such_variable"],
            id="unknown-variable",
        ),
        pytest.param(
            {
                "persons": {"a": {"salary": {PERIOD: 1.0}, "rent": {PERIOD: 1.0}}},
                "households": {"h": {"parents": ["a"]}},
            },
            ["persons", "a", "rent"],
            id="variable-of-another-entity",
        ),
        pytest.param(
            {
                "persons": {"a": {}, "b": {}},
                "households": {
                    "h": {
                        "parents": ["a"],
                        "housing_occupancy_status": {PERIOD: "castle"},
                    },
                    "g": {
                        "parents": ["b"],
                        "housing_occupancy_status": {PERIOD: "owner"},
                    },
                },
            },
            ["households", "h", "housing_occupancy_status"],
            id="rejected-value",
        ),
        pytest.param(
            {
                "persons": {
                    "a": {"salary": {"ETERNITY": 1.0}},
                    "b": {"salary": {PERIOD: 2.0}},
                },
                "households": {"h": {"parents": ["a", "b"]}},
            },
            ["persons", "a", "salary", "ETERNITY"],
            id="refused-period",
        ),
        pytest.param(
            {"salary": {PERIOD: 1.0}, "no_such_variable": {PERIOD: 1.0}},
            ["no_such_variable"],
            id="unknown-variable-without-entities",
        ),
    ],
)
def test_skipping_rejected_inputs_builds_the_situation_without_them(
    situation, rejected
):
    strict, skipping = _strict_and_skipping()

    with pytest.raises((SituationParsingError, VariableNotFoundError)):
        strict.build_from_dict(SYSTEM, copy.deepcopy(situation))
    simulation = skipping.build_from_dict(SYSTEM, copy.deepcopy(situation))
    expected = _built(_without(situation, rejected))

    assert simulation._user_input_keys == expected._user_input_keys
    assert simulation.to_input_dict() == expected.to_input_dict()
    assert_same(
        simulation, expected, outputs=["salary", "rent", "housing_occupancy_status"]
    )


def test_skipping_rejected_inputs_keeps_an_axis_over_an_unknown_variable():
    situation = {
        "persons": {"a": {"salary": {PERIOD: 1.0}}},
        "households": {"h": {"parents": ["a"]}},
        "axes": [
            [
                {"name": "no_such_variable", "count": 3, "min": 0, "max": 2},
                {"name": "salary", "count": 3, "min": 10, "max": 30, "period": PERIOD},
            ]
        ],
    }
    _, skipping = _strict_and_skipping()
    simulation = skipping.build_from_dict(SYSTEM, copy.deepcopy(situation))

    assert simulation.persons.count == 3
    np.testing.assert_array_equal(_values(simulation, "salary"), [10, 20, 30])


def test_the_builder_still_checks_entities_when_skipping_rejected_inputs():
    situation = {
        "persons": {"a": {}},
        "households": {"h": {"parents": ["a", "nobody"]}},
    }
    _, skipping = _strict_and_skipping()

    with pytest.raises(SituationParsingError):
        skipping.build_from_dict(SYSTEM, situation)


def test_building_a_situation_replaces_what_the_simulation_held():
    """What the old populations held, calculated or given, goes with them."""
    simulation = Simulation(situation=PARTIAL)
    simulation.calculate("income_tax", PERIOD)
    simulation.set_input("rent", PERIOD, [450.0])
    situation = {
        "persons": {"a": {}, "b": {}, "c": {"salary": {PERIOD: 100.0}}},
        "households": {"h": {"parents": ["a", "b"], "children": ["c"]}},
        "axes": [
            [{"name": "salary", "count": 2, "min": 0, "max": 2000, "period": PERIOD}]
        ],
    }

    simulation._build_from_situation(situation)
    expected = Simulation(situation=situation)

    assert_same(simulation, expected)
    assert simulation._user_input_keys == expected._user_input_keys
    assert sorted(simulation.input_variables) == sorted(expected.input_variables)
    assert simulation.has_axes == expected.has_axes
