"""A recursion over periods gives the same result whatever is cached.

``max_spiral_loops`` caps how many frames of one variable the stack holds.
A recursion longer than that used to be cut where the cap fell, returning
the default there. Since the cap counts uncached frames only, the result
depended on what had been calculated before: with an input of 1 in 2010 and
``recursive(t) = recursive(t - 1) + 1``, 2021 was 10 on a new simulation and
12 after calculating 2020.

Now a recursion heading towards an input or a period with no formula (an
anchor) finishes its deeper periods first and is evaluated all the way down.
One with no anchor is still cut, and nothing calculated from the cut value
is kept once the calculation ends.
"""

import pytest

from policyengine_core.errors import CycleError
from policyengine_core.simulations import simulation as simulation_module
from tests.fixtures.spirals import build, make_recurrence, run, stored_years


def fresh_and_after(variables, inputs, target, prior, loops=None):
    fresh = build(variables, inputs, loops)
    used = build(variables, inputs, loops)
    for request in prior:
        run(used, *request)
    return run(fresh, *target), run(used, *target)


def counting_up(start=2010):
    return make_recurrence("recursive", [("recursive", 1)], start=start)


@pytest.mark.parametrize(
    "loops, target, prior, expected",
    [(1, 2012, 2011, 3.0), (None, 2021, 2020, 12.0)],
)
def test_anchored_recursion_ignores_what_is_cached(loops, target, prior, expected):
    # The review's examples: an input of 1 in 2010 anchors the recursion.
    fresh, after = fresh_and_after(
        [counting_up()],
        {("recursive", 2010): 1},
        ("recursive", target),
        [("recursive", prior)],
        loops,
    )
    assert fresh == after == [expected]


def test_recursion_is_evaluated_down_to_the_first_formula():
    # No input: 2009 has no formula, so it is 0 and 2010 is 1.
    fresh, after = fresh_and_after(
        [counting_up()], {}, ("recursive", 2060), [("recursive", 2040)]
    )
    assert fresh == after == [51.0]


def test_recursion_is_evaluated_down_to_an_input():
    # An undated formula, anchored by the input in 2010 only.
    fresh, after = fresh_and_after(
        [counting_up(start=None)],
        {("recursive", 2010): 1},
        ("recursive", 2045),
        [("recursive", 2030)],
    )
    assert fresh == after == [36.0]


def test_forward_recursion_is_evaluated_up_to_the_end():
    ahead = make_recurrence("ahead", [("ahead", -1)], end=2030)
    fresh, after = fresh_and_after([ahead], {}, ("ahead", 2000), [("ahead", 2020)])
    assert fresh == after == [31.0]


def test_long_recursion_through_another_variable_is_evaluated_in_full():
    # first(t) = 1 + second(t), second(t) = 1 + first(t - 1); second's dated
    # formula anchors the chain.
    first = make_recurrence("first", [("second", 0)])
    second = make_recurrence("second", [("first", 1)], start=2010)
    fresh, after = fresh_and_after(
        [first, second], {}, ("first", 2040), [("first", 2030)]
    )
    # second(2009) = 0, first(2009) = 1; each later year adds 2.
    assert fresh == after == [1.0 + 2 * 31]


def test_unanchored_recursion_is_still_cut_at_the_limit():
    # No input and an undated formula: nothing ends the recursion, so it is
    # cut after max_spiral_loops frames, as before.
    unanchored = counting_up(start=None)
    fresh, after = fresh_and_after(
        [unanchored], {}, ("recursive", 2021), [("recursive", 2020)]
    )
    assert fresh == after == [10.0]


def test_values_reached_by_a_cut_are_not_kept():
    simulation = build([counting_up(start=None)])
    assert run(simulation, "recursive", 2021) == [10.0]
    assert stored_years(simulation, "recursive") == []


def test_values_above_a_cut_are_not_kept():
    # anchored(t) = 1 + unanchored(t) + anchored(t - 1): anchored's own chain
    # ends in 2009, but every value reads a cut recursion.
    unanchored = make_recurrence("unanchored", [("unanchored", 1)])
    anchored = make_recurrence(
        "anchored", [("unanchored", 0), ("anchored", 1)], start=2010
    )
    fresh, after = fresh_and_after(
        [unanchored, anchored],
        {},
        ("anchored", 2030),
        [("anchored", 2025), ("unanchored", 2028)],
    )
    assert fresh == after
    simulation = build([unanchored, anchored])
    run(simulation, "anchored", 2030)
    assert stored_years(simulation, "anchored") == []


def test_skipped_input_does_not_loop():
    # skip(t) = 1 + skip(t - 2) passes the 2011 input by when it starts from
    # an even year, so it can't end there; it is cut instead of retried.
    skip = make_recurrence("skip", [("skip", 2)])
    fresh, after = fresh_and_after(
        [skip], {("skip", 2011): 5}, ("skip", 2040), [("skip", 2030)]
    )
    assert fresh == after == [10.0]


def test_cycle_is_still_an_error():
    first = make_recurrence("first", [("second", 0)])
    second = make_recurrence("second", [("first", 0)])
    assert run(build([first, second]), "first", 2020) == CycleError.__name__


def test_formula_catching_exception_does_not_swallow_the_deeper_period():
    from policyengine_core.country_template.entities import Person
    from policyengine_core.periods import YEAR
    from policyengine_core.variables import Variable

    class guarded(Variable):
        value_type = float
        entity = Person
        definition_period = YEAR
        label = "Recursion inside try/except Exception"

        def formula_2010(person, period):
            try:
                return person("guarded", period.last_year) + 1
            except Exception:
                return person.filled_array(-1000.0)

    fresh, after = fresh_and_after(
        [guarded], {}, ("guarded", 2040), [("guarded", 2035)]
    )
    assert fresh == after == [31.0]


def test_cap_on_deeper_periods_falls_back_to_the_cut(monkeypatch):
    monkeypatch.setattr(simulation_module, "MAX_SPIRAL_DEFERRALS", 0)
    simulation = build([counting_up()], {("recursive", 2010): 1})
    assert run(simulation, "recursive", 2021) == [10.0]


def test_spiral_in_a_branch_is_purged_from_the_branch():
    simulation = build([counting_up(start=None)])
    branch = simulation.get_branch("other")
    assert run(branch, "recursive", 2021) == [10.0]
    assert stored_years(branch, "recursive", "other") == []
    assert run(branch, "recursive", 2022) == run(
        build([counting_up(start=None)]).get_branch("other"), "recursive", 2022
    )


def test_anchored_recursion_in_a_traced_branch_called_from_a_formula():
    from policyengine_core.country_template.entities import Person
    from policyengine_core.periods import YEAR
    from policyengine_core.variables import Variable

    class from_branch(Variable):
        value_type = float
        entity = Person
        definition_period = YEAR
        label = "Reads recursive on a branch"

        def formula(person, period):
            branch = person.simulation.get_branch("other")
            return branch.calculate("recursive", period)

    traced = build([counting_up(), from_branch], {("recursive", 2010): 1})
    traced.trace = True
    plain = build([counting_up(), from_branch], {("recursive", 2010): 1})
    assert run(traced, "from_branch", 2040) == run(plain, "from_branch", 2040)
    assert run(plain, "from_branch", 2040) == [31.0]


def test_traced_branch_cut_from_a_formula_is_purged():
    from policyengine_core.country_template.entities import Person
    from policyengine_core.periods import YEAR
    from policyengine_core.variables import Variable

    class from_branch(Variable):
        value_type = float
        entity = Person
        definition_period = YEAR
        label = "Reads recursive on a branch"

        def formula(person, period):
            branch = person.simulation.get_branch("other")
            return branch.calculate("recursive", period)

    simulation = build([counting_up(start=None), from_branch])
    simulation.trace = True
    run(simulation, "from_branch", 2021)
    branch = simulation.branches["other"]
    assert stored_years(branch, "recursive", "other") == []
    assert stored_years(simulation, "from_branch") == []


def test_branch_made_after_a_cut_does_not_keep_the_cut_values():
    from policyengine_core.country_template.entities import Person
    from policyengine_core.periods import YEAR
    from policyengine_core.variables import Variable

    class then_branch(Variable):
        value_type = float
        entity = Person
        definition_period = YEAR
        label = "Reads a cut recursion, then makes a branch"

        def formula(person, period):
            value = person("recursive", period)
            person.simulation.get_branch("late")
            return value

    simulation = build([counting_up(start=None), then_branch])
    run(simulation, "then_branch", 2021)
    branch = simulation.branches["late"]
    assert stored_years(branch, "recursive", "default") == []


def test_untraced_branch_cut_from_a_formula_is_purged_from_the_caller():
    from policyengine_core.country_template.entities import Person
    from policyengine_core.periods import YEAR
    from policyengine_core.variables import Variable

    class from_branch(Variable):
        value_type = float
        entity = Person
        definition_period = YEAR
        label = "Reads recursive on a branch"

        def formula(person, period):
            branch = person.simulation.get_branch("other")
            return branch.calculate("recursive", period)

    simulation = build([counting_up(start=None), from_branch])
    assert run(simulation, "from_branch", 2021) == [10.0]
    assert stored_years(simulation, "from_branch") == []
    assert stored_years(simulation.branches["other"], "recursive", "other") == []


def test_branch_spiral_does_not_purge_the_parent():
    # A cut in a branch marks values for purging in that branch only. A
    # parent sharing the branch's set would later delete its own values at
    # those periods, an input set after branching included.
    simulation = build([counting_up(start=None)])
    branch = simulation.get_branch("other")
    simulation.set_input("recursive", "2015", [100])
    assert run(branch, "recursive", 2021) == [10.0]
    assert run(simulation, "recursive", 2016) == [101.0]
    assert simulation.get_array("recursive", "2015").tolist() == [100.0]
