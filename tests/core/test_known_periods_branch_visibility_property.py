"""``Holder.get_known_periods(branch_name)`` agrees with ``get_array`` for any
stored values.

Random values are stored, in memory or on disk, as inputs or derived, under
the branches of one lineage: ``default``, its branch ``a``, ``a``'s branch
``a_b``, ``default``'s other branch ``c``, and a branch named ``default``
nested under ``a``. Each branch of the lineage, reading its own holder, sees:

* ``get_known_periods(branch_name)`` lists each period ``get_array(period,
  branch_name)`` reads a stored value for, once, and no other period: the
  listing against the per-period reference, ``get_array``;
* ``get_input_periods(branch_name)`` lists the periods it lists whose value
  is an input (``not is_derived(period, branch_name)``): the one pass over
  the stored keys against the per-period reference.

``test_known_periods_branch_visibility.py`` pins the same with examples.
"""

import os
import tempfile

import numpy as np
import pytest

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.country_template.entities import Person
from policyengine_core.model_api import YEAR, Variable
from policyengine_core.simulations import SimulationBuilder

STORED_BRANCHES = ("default", "a", "a_b", "c", "unrelated")
YEARS = (2015, 2016, 2017)


class yearly_income(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
    label = "Yearly income"


@pytest.fixture(scope="module")
def system():
    system = CountryTaxBenefitSystem()
    system.add_variable(yearly_income)
    return system


def lineage(system):
    simulation = SimulationBuilder().build_default_simulation(system, count=1)
    a = simulation.get_branch("a")
    return [
        simulation,
        a,
        a.get_branch("a_b"),
        simulation.get_branch("c"),
        a.get_branch("default"),
    ]


stored_value = st.tuples(
    st.sampled_from(STORED_BRANCHES),
    st.sampled_from(YEARS),
    st.booleans(),  # derived
    st.booleans(),  # on disk
)


@hypothesis.settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
@hypothesis.given(stored=st.lists(stored_value, max_size=12))
def test_known_periods_for_a_branch_are_the_periods_it_reads(system, stored):
    with tempfile.TemporaryDirectory() as directory:
        for index, reader in enumerate(lineage(system)):
            holder = reader.get_holder("yearly_income")
            reader_directory = os.path.join(directory, str(index))
            os.mkdir(reader_directory)
            holder._disk_storage = holder.create_disk_storage(
                reader_directory, preserve=True
            )
            for value, (branch_name, year, derived, on_disk) in enumerate(stored):
                storage = holder._disk_storage if on_disk else holder._memory_storage
                storage.put(
                    np.array([float(value)]),
                    periods.period(year),
                    branch_name,
                    derived=derived,
                )

            branch_name = reader.branch_name
            listed = holder.get_known_periods(branch_name)
            read = [
                period
                for period in map(periods.period, YEARS)
                if holder.get_array(period, branch_name) is not None
            ]

            assert len(listed) == len(set(listed))
            assert set(listed) == set(read)
            assert set(holder.get_input_periods(branch_name)) == {
                period
                for period in listed
                if not holder.is_derived(period, branch_name)
            }
