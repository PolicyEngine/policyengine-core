"""A person-level DataFrame loads, exports and reloads without surprises."""

import logging

import numpy as np
import pandas as pd
import pytest

from policyengine_core.data import Dataset
from policyengine_core.entities import build_entity
from policyengine_core.periods import ETERNITY, YEAR
from policyengine_core.simulations import Simulation
from policyengine_core.taxbenefitsystems import TaxBenefitSystem
from policyengine_core.variables import Variable

Person = build_entity(key="person", plural="persons", label="Person", is_person=True)
Household = build_entity(
    key="household",
    plural="households",
    label="Household",
    roles=[{"key": "member", "plural": "members", "label": "Members"}],
)
Family = build_entity(
    key="family",
    plural="families",
    label="Family",
    roles=[{"key": "member", "plural": "members", "label": "Members"}],
)


def person_variable(name, value_type=float, period=YEAR, entity=Person, **attributes):
    return type(
        name,
        (Variable,),
        {
            "value_type": value_type,
            "entity": entity,
            "definition_period": period,
            "label": name,
            **attributes,
        },
    )


def formula_of_ages(person, period):
    return person("age", period) * 0


class Countable(TaxBenefitSystem):
    source_input_variables = frozenset({"observed_role"})


def build(data, time_period=2024, system_class=Countable):
    system = system_class([Person, Household, Family])
    system.add_variable(person_variable("household_id", int, entity=Household))
    system.add_variable(person_variable("family_id", int, entity=Family))
    system.add_variable(person_variable("person_household_id", int))
    system.add_variable(person_variable("person_family_id", int))
    system.add_variable(person_variable("age"))
    system.add_variable(person_variable("is_head", bool, ETERNITY))
    system.add_variable(
        person_variable("observed_role", bool, ETERNITY, formula=formula_of_ages)
    )
    system.add_variable(
        person_variable("saved_result", formula=formula_of_ages),
    )
    system.add_variable(
        person_variable(
            "person_id",
            int,
            formula=lambda person, period: np.arange(person.count),
        )
    )
    return Simulation(
        tax_benefit_system=system,
        dataset=Dataset.from_dataframe(data, time_period),
    )


def people():
    return pd.DataFrame(
        {
            "household_id": [1, 1, 2],
            "family_id": [1, 1, 2],
            "age": [35.0, 5.0, 60.0],
        }
    )


def test_rows_number_people_when_no_column_does():
    simulation = build(people())
    assert simulation.persons.ids.tolist() == [0, 1, 2]


def test_a_person_id_column_with_a_period_keeps_its_values():
    data = people().assign(person_id__2024=[10, 20, 30])
    assert build(data).persons.ids.tolist() == [10, 20, 30]


def test_group_ids_on_person_rows_declare_the_groups():
    simulation = build(people())
    assert simulation.populations["household"].count == 2
    assert simulation.populations["family"].count == 2


@pytest.mark.parametrize(
    "missing, named",
    [
        ("household_id", "person_household_id or household_id"),
        ("family_id", "person_family_id or family_id"),
    ],
)
def test_a_missing_group_column_names_the_fix(missing, named):
    with pytest.raises(ValueError, match=named):
        build(people().drop(columns=missing))


def test_a_column_name_with_two_separators_loads_as_an_unknown_column(caplog):
    data = people().assign(**{"survey__note__2024": ["a", "b", "c"]})
    with caplog.at_level(logging.WARNING):
        simulation = build(data)
    assert simulation.calculate("age", 2024).tolist() == [35.0, 5.0, 60.0]
    assert "survey__note__2024" in caplog.text


@pytest.mark.parametrize(
    "column", ["is_head", "is_head__2024", "is_head__ETERNITY", "is_head__2019"]
)
def test_an_all_time_variable_loads_from_any_column_spelling(column):
    simulation = build(people().assign(**{column: [True, False, True]}))
    assert simulation.calculate("is_head", 2030).tolist() == [True, False, True]


def test_an_export_spells_an_all_time_variable_eternity():
    simulation = build(people().assign(is_head=[True, False, True]))
    assert "is_head__ETERNITY" in simulation.to_input_dataframe().columns


def test_a_period_export_keeps_that_period_and_all_time_variables():
    data = people().assign(
        age__2025=[36.0, 6.0, 61.0],
        is_head=[True, False, True],
    )
    columns = set(build(data).to_input_dataframe(period=2025).columns)
    assert "age__2025" in columns
    assert "is_head__ETERNITY" in columns
    assert "age__2024" not in columns


def test_a_period_export_reloads_on_its_own():
    data = people().assign(age__2025=[36.0, 6.0, 61.0], is_head=[True, False, True])
    exported = build(data).to_input_dataframe(period=2025)
    reloaded = build(exported, time_period=2025)
    assert reloaded.calculate("age", 2025).tolist() == [36.0, 6.0, 61.0]
    assert reloaded.calculate("is_head", 2025).tolist() == [True, False, True]


def test_a_declared_source_input_survives_the_round_trip():
    data = people().assign(observed_role=[True, False, True])
    simulation = build(data)
    assert simulation.calculate("observed_role", 2024).tolist() == [True, False, True]
    reloaded = build(simulation.to_input_dataframe())
    assert reloaded.calculate("observed_role", 2024).tolist() == [True, False, True]
    assert "observed_role" in simulation.true_input_variables


def test_population_ids_survive_the_round_trip_when_they_have_formulas():
    data = people().assign(person_id=[10, 20, 30])
    exported = build(data).to_input_dataframe()
    assert exported["person_id__2024"].tolist() == [10, 20, 30]
    assert build(exported).persons.ids.tolist() == [10, 20, 30]


def test_an_undeclared_formula_value_stays_out_and_the_export_says_so(caplog):
    simulation = build(people().assign(saved_result=[9.0, 9.0, 9.0]))
    with caplog.at_level(logging.WARNING):
        exported = simulation.to_input_dataframe()
    assert "saved_result__2024" not in exported.columns
    assert "saved_result" in caplog.text
    assert build(exported).calculate("saved_result", 2024).tolist() == [0.0] * 3


def test_a_clean_export_warns_about_nothing(caplog):
    simulation = build(people())
    with caplog.at_level(logging.WARNING):
        simulation.to_input_dataframe()
    assert caplog.text == ""


def test_exporting_computed_variables_warns_about_nothing(caplog):
    simulation = build(people().assign(saved_result=[9.0, 9.0, 9.0]))
    with caplog.at_level(logging.WARNING):
        exported = simulation.to_input_dataframe(include_computed_variables=True)
    assert "saved_result__2024" in exported.columns
    assert caplog.text == ""
