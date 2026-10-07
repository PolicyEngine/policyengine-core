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


def test_recursion_changing_direction_ends():
    # zigzag(t) = 1 + zigzag(t + 2) in odd years and 1 + zigzag(t - 1) in
    # even ones, until 2020: it heads back towards its first formula, then
    # forward towards its end, in turn. Each deeper period is started once,
    # so the calculation ends, and with the same value however it starts.
    from policyengine_core.country_template.entities import Person
    from policyengine_core.periods import YEAR
    from policyengine_core.variables import Variable

    class zigzag(Variable):
        value_type = float
        entity = Person
        definition_period = YEAR
        label = "Recursion heading back and forward in turn"
        end = "2020-12-31"

        def formula_2010(person, period):
            lag = 1 if period.start.year % 2 == 0 else -2
            return person("zigzag", period.offset(-lag, "year")) + 1

    # Evaluated in full: 2021 has no formula and is 0, 2009 likewise.
    expected = {2010: 1, 2011: 5, 2012: 6, 2013: 4, 2014: 5, 2015: 3}
    expected.update({2016: 4, 2017: 2, 2018: 3, 2019: 1, 2020: 2})
    for loops in (1, 2, 10):
        for year, value in expected.items():
            fresh, after = fresh_and_after(
                [zigzag], {}, ("zigzag", year), [("zigzag", 2017)], loops
            )
            assert fresh == after == [float(value)]


def test_copy_made_in_a_formula_ends_its_recursion():
    # A formula that makes a copy of the simulation and calculates on it
    # makes a new copy each time the calculation starts again, so the copy's
    # deeper periods are never finished first: its recursion is cut, as
    # before, instead of starting again for ever.
    from policyengine_core.country_template.entities import Person
    from policyengine_core.periods import YEAR
    from policyengine_core.variables import Variable

    class on_copy(Variable):
        value_type = float
        entity = Person
        definition_period = YEAR
        label = "Calculates recursive on a new copy of the simulation"

        def formula(person, period):
            return person.simulation.clone().calculate("recursive", period)

    simulation = build([counting_up(), on_copy], {}, 1)
    assert run(simulation, "on_copy", 2015) == [1.0]
    # On the copy itself, outside a calculation, it is evaluated in full.
    copy = simulation.clone()
    assert run(copy, "recursive", 2015) == [6.0]


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


# Round-2 review regressions. Each case compares a calculation with the same
# calculation in a new simulation, or with the full evaluation of an anchored
# chain, which is what a new simulation returns.


def reads(name, target, lag=0):
    """A yearly variable equal to ``target`` ``lag`` years earlier."""
    from policyengine_core.country_template.entities import Person
    from policyengine_core.periods import YEAR
    from policyengine_core.variables import Variable

    def formula(person, period):
        return person(target, period.offset(-lag, "year"))

    return type(
        name,
        (Variable,),
        dict(
            value_type=float,
            entity=Person,
            definition_period=YEAR,
            label=f"Reads {target}",
            formula=formula,
        ),
    )


@pytest.mark.parametrize("traced", [False, True])
def test_cut_in_the_parent_keeps_a_branch_input_under_the_same_name(traced):
    # A branch of a branch can be named "default", like the simulation it
    # descends from. A cut in the simulation's calculation used to delete the
    # simulation's branch-name key from every descendant, and with it this
    # branch's own input.
    simulation = build(
        [counting_up(start=None), reads("wrapper", "recursive")], max_spiral_loops=1
    )
    simulation.trace = traced
    branch = simulation.get_branch("one").get_branch("default")
    branch.set_input("wrapper", "2021", [100])
    assert run(simulation, "wrapper", 2021) == [1.0]
    assert branch.get_array("wrapper", "2021").tolist() == [100.0]
    assert run(branch, "wrapper", 2021) == [100.0]


@pytest.mark.parametrize("traced", [False, True])
def test_cut_keeps_an_input_set_on_a_branch_during_the_calculation(traced):
    from policyengine_core.country_template.entities import Person
    from policyengine_core.periods import YEAR
    from policyengine_core.variables import Variable

    class via(Variable):
        value_type = float
        entity = Person
        definition_period = YEAR
        label = "Sets an input on a branch, then reads a cut recursion there"

        def formula(person, period):
            branch = person.simulation.get_branch("one").get_branch("default")
            branch.set_input("via", period, [100])
            return branch.calculate("recursive", period)

    simulation = build([counting_up(start=None), via], max_spiral_loops=1)
    simulation.trace = traced
    assert run(simulation, "via", 2021) == [1.0]
    branch = simulation.branches["one"].branches["default"]
    assert branch.get_array("via", "2021").tolist() == [100.0]
    assert run(branch, "via", 2021) == [100.0]


def catching(name, handler):
    """A dated recursion whose formula reads last year inside a bare except."""
    from policyengine_core.country_template.entities import Person
    from policyengine_core.periods import YEAR
    from policyengine_core.variables import Variable

    def formula_2010(person, period):
        try:
            return person(name, period.last_year) + 1
        except:  # noqa: E722 - the case under test
            return handler(person, period)

    return type(
        name,
        (Variable,),
        dict(
            value_type=float,
            entity=Person,
            definition_period=YEAR,
            label="Recursion inside a bare except",
            formula_2010=formula_2010,
        ),
    )


def _fallback(person, period):
    return person.filled_array(-1000.0)


def _reraise(person, period):
    raise ValueError("the formula turned the exception into its own")


def _fallback_calculation(person, period):
    return person("plain", period) - 1000


@pytest.mark.parametrize("handler", [_fallback, _reraise, _fallback_calculation])
@pytest.mark.parametrize("prior", [[], [2011], [2010, 2011]])
def test_bare_except_in_a_formula_does_not_swallow_the_deeper_period(handler, prior):
    # The chain is anchored (2009 has no formula), so 2012 is 3 whatever was
    # calculated before. A bare except used to catch the request to finish
    # the deeper period first and cache its fallback.
    variables = [catching("catches_all", handler), make_recurrence("plain", [])]
    fresh, after = fresh_and_after(
        variables,
        {},
        ("catches_all", 2012),
        [("catches_all", year) for year in prior],
        loops=1,
    )
    assert fresh == after == [3.0]


def requires_caller():
    from policyengine_core.country_template.entities import Person
    from policyengine_core.periods import YEAR
    from policyengine_core.variables import Variable

    class caller(Variable):
        value_type = float
        entity = Person
        definition_period = YEAR
        label = "Calls dependent"

        def formula(person, period):
            return person("dependent", period)

    class dependent(Variable):
        value_type = float
        entity = Person
        definition_period = YEAR
        label = "Requires caller to be requested first"
        requires_computation_after = "caller"

        def formula_2010(person, period):
            return person("dependent", period.last_year) + 1

    return [caller, dependent]


@pytest.mark.parametrize(
    "loops, year, prior, expected",
    [
        (1, 2012, [], 3.0),
        (1, 2012, [2011], 3.0),
        (None, 2021, [], 12.0),
        (None, 2021, [2015], 12.0),
    ],
)
def test_deeper_periods_keep_the_caller_requires_computation_after_needs(
    loops, year, prior, expected
):
    # dependent may only be calculated while caller is: the deeper periods
    # finished first used to run with an empty stack, and raised ValueError.
    fresh, after = fresh_and_after(
        requires_caller(),
        {},
        ("caller", year),
        [("caller", earlier) for earlier in prior],
        loops,
    )
    assert fresh == after == [expected]


def test_requires_computation_after_still_refuses_a_direct_request():
    assert run(build(requires_caller()), "dependent", 2012) == ValueError.__name__


def copying(copies, branch=False):
    """A variable that reads recursive, then copies the simulation."""
    from policyengine_core.country_template.entities import Person
    from policyengine_core.periods import YEAR
    from policyengine_core.variables import Variable

    def formula(person, period):
        value = person("recursive", period)
        simulation = person.simulation
        if branch:
            copies.append(simulation.get_branch(f"late{len(copies)}"))
        else:
            copies.append(simulation.clone())
        return value

    return type(
        "copy_after_cut",
        (Variable,),
        dict(
            value_type=float,
            entity=Person,
            definition_period=YEAR,
            label="Copies the simulation after a cut",
            formula=formula,
        ),
    )


@pytest.mark.parametrize("branch", [False, True])
@pytest.mark.parametrize("loops", [1, None])
@pytest.mark.parametrize("year", [2015, 2021, 2022])
def test_copy_made_during_a_cut_does_not_keep_the_cut_values(branch, loops, year):
    # The copy is made while the cut recursion's values are cached and
    # waiting to be purged; it used to keep them, so its later results
    # depended on the cut.
    copies = []
    simulation = build([counting_up(start=None), copying(copies, branch)], {}, loops)
    run(simulation, "copy_after_cut", 2021)
    copy = copies[0]
    name = copy.branch_name
    assert stored_years(copy, "recursive", "default") == []
    fresh = build([counting_up(start=None)], {}, loops)
    if branch:
        fresh = fresh.get_branch(name)
    assert run(copy, "recursive", year) == run(fresh, "recursive", year)


@pytest.mark.parametrize("loops", [1, 2, 10])
@pytest.mark.parametrize("end", [2012, 2020])
@pytest.mark.parametrize("carry", [False, True])
def test_forward_anchor_does_not_read_carried_calculations(carry, end, loops):
    # forward(t) = forward(t + 1) + 1 until its end; 2008 and 2009 are 7.
    # Only inputs carry over, so the year after the end is 0 whatever was
    # calculated before.
    from policyengine_core.country_template.entities import Person
    from policyengine_core.periods import YEAR
    from policyengine_core.variables import Variable

    class forward(Variable):
        value_type = float
        entity = Person
        definition_period = YEAR
        label = "Forward recursion with an earlier constant formula"

        def formula_2008(person, period):
            return person.filled_array(7)

        def formula_2010(person, period):
            return person("forward", period.offset(1, "year")) + 1

    forward.end = f"{end}-12-31"

    def new():
        simulation = build([forward], {}, loops)
        simulation.tax_benefit_system.auto_carry_over_input_variables = carry
        return simulation

    fresh, used = new(), new()
    run(used, "forward", 2008)
    assert run(fresh, "forward", 2010) == run(used, "forward", 2010)
    assert run(fresh, "forward", 2010) == [end - 2010 + 1.0]


@pytest.mark.parametrize("loops, year", [(1, 2011), (None, 2021)])
def test_traced_deeper_periods_serialize_the_completed_calculation(loops, year):
    # The first attempt at the request is abandoned when it reaches the
    # limit; its trace nodes have no value. The flat trace used to keep them.
    simulation = build([counting_up()], {}, loops)
    simulation.trace = True
    expected = [year - 2009.0]
    assert run(simulation, "recursive", year) == expected
    trace = simulation.tracer.get_serialized_flat_trace()
    for earlier in range(2010, year + 1):
        node = trace[f"recursive<{earlier}, (default)>"]
        assert node["value"] == [earlier - 2009.0]
    assert trace[f"recursive<{year}, (default)>"]["dependencies"] == [
        f"recursive<{year - 1}, (default)>"
    ]
    assert simulation.tracer.stack == []


def without_storage(simulation, method):
    if method == "drop":
        simulation.get_holder("recursive")._do_not_store = True
    elif method == "blacklist":
        simulation.opt_out_cache = True
        simulation.tax_benefit_system.cache_blacklist = {"recursive"}


@pytest.mark.parametrize("traced", [False, True])
@pytest.mark.parametrize("method", ["drop", "blacklist"])
@pytest.mark.parametrize(
    "start, loops, year, prior, expected",
    [
        # Anchored: evaluated in full, whatever the storage or tracing.
        (2010, 1, 2012, [], 3.0),
        (2010, 1, 2012, [2011], 3.0),
        (2010, None, 2021, [], 12.0),
        (2010, None, 2021, [2015], 12.0),
        # Not anchored: cut at the limit; nothing calculated from the cut is
        # reused afterwards, even untraced from the fast cache.
        (None, 1, 2012, [], 1.0),
        (None, 1, 2012, [2011], 1.0),
        (None, None, 2021, [], 10.0),
        (None, None, 2021, [2020], 10.0),
    ],
)
def test_deeper_periods_do_not_depend_on_storage_or_tracing(
    traced, method, start, loops, year, prior, expected
):
    def new():
        simulation = build([counting_up(start)], {}, loops)
        simulation.trace = traced
        without_storage(simulation, method)
        return simulation

    fresh, used = new(), new()
    for earlier in prior:
        run(used, "recursive", earlier)
    assert run(fresh, "recursive", year) == [expected]
    assert run(used, "recursive", year) == [expected]


def test_long_anchored_recursion_is_evaluated_in_full_however_long():
    # 10,002 monthly formula periods at max_spiral_loops 1: past what a
    # budget of 10,000 deeper periods allowed, after which the chain was
    # cut, or not, depending on what was cached.
    from policyengine_core.country_template.entities import Person
    from policyengine_core.periods import MONTH
    from policyengine_core.variables import Variable

    class monthly(Variable):
        value_type = float
        entity = Person
        definition_period = MONTH
        label = "Dated monthly backward recurrence"

        def formula_2010(person, period):
            return person("monthly", period.offset(-1, "month")) + 1

    fresh = build([monthly], {}, 1)
    warm = build([monthly], {}, 1)
    assert warm.calculate("monthly", "2010-01").tolist() == [1.0]
    assert fresh.calculate("monthly", "2843-06").tolist() == [10002.0]
    assert warm.calculate("monthly", "2843-06").tolist() == [10002.0]


@pytest.mark.parametrize("spiral_first", [False, True])
def test_explicit_invalidation_keeps_precedence_over_spiral_cleanup(spiral_first):
    from policyengine_core.periods import period

    simulation = build([counting_up()], {("recursive", 2021): 40})
    key = ("recursive", period("2021"))
    markers = [
        simulation._invalidate_spiral_cache_entry,
        simulation.invalidate_cache_entry,
    ]
    if not spiral_first:
        markers.reverse()
    for mark in markers:
        mark(*key)
    simulation.purge_cache_of_invalid_values()
    assert simulation.get_array(*key) is None


def test_spiral_cleanup_and_copy_preserve_an_input_at_the_marked_period():
    from policyengine_core.periods import period

    simulation = build([counting_up()], {("recursive", 2021): 40})
    key = ("recursive", period("2021"))
    simulation._invalidate_spiral_cache_entry(*key)
    copy = simulation.clone()
    assert copy.invalidated_caches == set()
    assert copy.get_array(*key).tolist() == [40]
    simulation.purge_cache_of_invalid_values()
    assert simulation.get_array(*key).tolist() == [40]


def test_explicit_invalidation_removes_contained_inputs_and_fast_values():
    from policyengine_core.periods import period

    simulation = build([])
    simulation.set_input("salary", "2020-06", [100])
    assert simulation.calculate("salary", "2020-06").tolist() == [100]
    simulation.invalidate_cache_entry("salary", period("2020"))
    simulation.purge_cache_of_invalid_values()
    assert simulation.get_array("salary", "2020-06") is None
    assert simulation.calculate("salary", "2020-06").tolist() == [0]
