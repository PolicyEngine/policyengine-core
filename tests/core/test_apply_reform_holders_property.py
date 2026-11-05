"""Check holder identity, defaults and recorded inputs after a reform."""

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core import periods
from policyengine_core.reforms import Reform
from policyengine_core.variables import Variable
from tests.fixtures.reform_holders import YEAR, build, change_default, store_on_disk

pytestmark = pytest.mark.filterwarnings(
    "ignore:Memory configuration:policyengine_core.warnings.MemoryConfigWarning"
)

PROPERTY_SETTINGS = hypothesis.settings(max_examples=50, deadline=None, database=None)
MASKS = st.lists(st.booleans(), min_size=1, max_size=6)
DEFAULTS = st.integers(min_value=-1_000, max_value=1_000)
REPLACED = ("award", "annual_value", "group_value")


def assert_holder_identity(simulation):
    for population in simulation.populations.values():
        for name, holder in population._holders.items():
            assert holder.variable is simulation.tax_benefit_system.variables[name]


def expected_award(mask, default):
    return np.where(mask, 100.0, float(default))


def change_several_defaults(default):
    class ChangeSeveralDefaults(Reform):
        def apply(self):
            for offset, name in enumerate(REPLACED):
                attributes = {"default_value": float(default + offset)}
                if name == "award":
                    attributes["defined_for"] = "eligible"
                self.update_variable(type(name, (Variable,), attributes))

    return ChangeSeveralDefaults


@PROPERTY_SETTINGS
@hypothesis.given(
    mask=MASKS,
    default=DEFAULTS,
    order=st.permutations(REPLACED),
    warm=st.booleans(),
    branches=st.sampled_from(("none", "shared", "independent", "both")),
)
@hypothesis.example(
    mask=[False, True],
    default=7,
    order=("group_value", "award", "annual_value"),
    warm=True,
    branches="both",
)
def test_every_existing_holder_uses_its_system_variable(
    mask, default, order, warm, branches
):
    simulation = build(mask)
    for name in REPLACED:
        population = simulation.get_variable_population(name)
        del population._holders[name]
    for name in order:
        simulation.get_holder(name)
        if warm:
            simulation.calculate(name, YEAR)
    related = []
    if branches in ("shared", "both"):
        related.append((simulation.get_branch("shared"), default))
    if branches in ("independent", "both"):
        related.append((simulation.get_branch("independent", clone_system=True), 0))

    simulation.apply_reform(change_several_defaults(default))

    for member in [simulation, *(branch for branch, _ in related)]:
        assert_holder_identity(member)
    for member, expected_default in [(simulation, default), *related]:
        np.testing.assert_array_equal(
            member.calculate("award", YEAR), expected_award(mask, expected_default)
        )
    np.testing.assert_array_equal(
        simulation.calculate("annual_value", YEAR),
        np.full(len(mask), default + 1),
    )
    np.testing.assert_array_equal(
        simulation.calculate("group_value", YEAR), [default + 2]
    )


@PROPERTY_SETTINGS
@hypothesis.given(mask=MASKS, default=DEFAULTS, warm=st.booleans())
@hypothesis.example(mask=[False, False], default=7, warm=False)
@hypothesis.example(mask=[False, True], default=7, warm=True)
def test_person_result_does_not_depend_on_other_people(mask, default, warm):
    combined = build(mask)
    if warm:
        combined.calculate("award", YEAR)
    reform = change_default(default)
    combined.apply_reform(reform)
    together = combined.calculate("award", YEAR)
    alone = []
    for eligible in mask:
        individual = build([eligible])
        if warm:
            individual.calculate("award", YEAR)
        individual.apply_reform(reform)
        alone.append(individual.calculate("award", YEAR)[0])

    np.testing.assert_array_equal(together, expected_award(mask, default))
    np.testing.assert_array_equal(alone, expected_award(mask, default))
    np.testing.assert_array_equal(together, alone)


@PROPERTY_SETTINGS
@hypothesis.given(mask=MASKS, default=DEFAULTS, warm=st.booleans())
@hypothesis.example(mask=[False, False], default=7, warm=True)
@hypothesis.example(mask=[False, True], default=7, warm=False)
def test_reform_before_or_after_holder_creation_returns_same_values(
    mask, default, warm
):
    before_holder = build(mask)
    del before_holder.person._holders["award"]
    after_holder = build(mask)
    if warm:
        after_holder.calculate("award", YEAR)
    reform = change_default(default)

    before_holder.apply_reform(reform)
    after_holder.apply_reform(reform)

    expected = expected_award(mask, default)
    np.testing.assert_array_equal(before_holder.calculate("award", YEAR), expected)
    np.testing.assert_array_equal(after_holder.calculate("award", YEAR), expected)
    assert_holder_identity(before_holder)
    assert_holder_identity(after_holder)


INPUT_ROWS = st.lists(
    st.tuples(
        st.booleans(),
        st.integers(min_value=-10_000, max_value=10_000),
        st.integers(min_value=-10_000, max_value=10_000),
    ),
    min_size=1,
    max_size=6,
)


@PROPERTY_SETTINGS
@hypothesis.given(
    rows=INPUT_ROWS,
    defaults=st.lists(DEFAULTS, min_size=1, max_size=4),
    on_disk=st.booleans(),
    branch=st.booleans(),
)
@hypothesis.example(
    rows=[(False, 12, 34), (True, -5, 9)],
    defaults=[7, 11],
    on_disk=True,
    branch=True,
)
def test_unchanged_recorded_inputs_survive_reforms(rows, defaults, on_disk, branch):
    mask, incomes, eternal = zip(*rows)
    simulation = build(mask)
    if on_disk:
        store_on_disk(simulation, "income", "annual_value", "eternal_value")
    supplied = {
        "income": np.asarray(incomes, dtype=np.float32),
        "annual_value": np.asarray(incomes, dtype=np.float32) + 1,
        "eternal_value": np.asarray(eternal, dtype=np.float32),
    }
    for name, values in supplied.items():
        simulation.set_input(name, YEAR, values)
        if on_disk:
            holder = simulation.get_holder(name)
            assert holder._disk_storage.get(periods.period(YEAR)) is not None
            assert holder._memory_storage.get(periods.period(YEAR)) is None
    members = [simulation]
    if branch:
        members.append(simulation.get_branch("existing"))
    input_keys = [set(member._user_input_keys) for member in members]

    for default in defaults:
        simulation.apply_reform(change_default(default))
        for member, keys in zip(members, input_keys):
            assert member._user_input_keys == keys
            for name, expected in supplied.items():
                np.testing.assert_array_equal(member.calculate(name, YEAR), expected)
            np.testing.assert_array_equal(
                member.calculate("award", YEAR), expected_award(mask, default)
            )
            assert_holder_identity(member)


@PROPERTY_SETTINGS
@hypothesis.given(mask=MASKS, default=DEFAULTS, warm=st.booleans())
@hypothesis.example(mask=[False, False], default=7, warm=True)
@hypothesis.example(mask=[False, True], default=7, warm=False)
def test_applying_same_reform_object_twice_returns_same_values(mask, default, warm):
    once = build(mask)
    twice = build(mask)
    inputs = np.arange(len(mask), dtype=np.float32) + 5
    for simulation in (once, twice):
        simulation.set_input("income", YEAR, inputs)
        if warm:
            simulation.calculate("award", YEAR)
    reform = change_default(default)

    once.apply_reform(reform)
    twice.apply_reform(reform)
    twice.calculate("award", YEAR)
    twice.apply_reform(reform)

    expected = expected_award(mask, default)
    for simulation in (once, twice):
        np.testing.assert_array_equal(simulation.calculate("award", YEAR), expected)
        np.testing.assert_array_equal(simulation.calculate("income", YEAR), inputs)
        assert_holder_identity(simulation)
    np.testing.assert_array_equal(
        twice.calculate("award", YEAR), once.calculate("award", YEAR)
    )
