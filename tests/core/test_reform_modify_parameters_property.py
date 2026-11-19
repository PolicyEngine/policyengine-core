"""Properties of ``modify_parameters``.

For any baseline read at any instants, and any reform whose ``apply()`` edits
the tree directly, reads it, calls modifiers ending in any way and uses the dict
form, on either path a reform runs:

* Fresh reads: once ``modify_parameters`` has run last, the system, its tree and
  each bracket read as a tree never read at all, given the same edits. That
  holds for a modifier that raises as well.
* The two paths agree: an instance and ``apply_reform_set`` leave the same tree.
* The baseline stays put: an instance changes nothing in its baseline, its
  values, its metadata or the nodes it has cached.
* Composition: two modifiers do what one modifier making both sets of edits does.
* Differential: for one modifier on a baseline never read, the tree matches what
  the deep copy in ``Reform.modify_parameters`` used to leave. Only this case
  can match: the deep copy threw away earlier steps, and read stale values.

Two things are outside these properties, both on purpose. A direct edit to a tree
already read, with no ``modify_parameters`` after it, can read stale: #621 is
where that gets settled. A modifier that edits nested metadata or a list value in
place reaches the baseline, as ``modify_parameters`` documents: the edits here
replace top-level metadata keys only, and the tree has no list values.

Examples are in ``test_reform_modify_parameters.py``.
"""

import copy

import pytest

# The smoke job installs Core without the dev extra but collects every module.
pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from policyengine_core.parameters import ParameterNode  # noqa: E402
from policyengine_core.reforms import Reform  # noqa: E402
from tests.fixtures.parameter_metadata_clones import INSTANTS, snapshot  # noqa: E402
from tests.fixtures.reform_modifiers import (  # noqa: E402
    ModifierFailed,
    apply_steps,
    assert_keeps_its_cached_nodes,
    cached_nodes,
    modifier,
    new_system,
    reads,
    run_as_instance,
    run_in_place,
    system_reads,
    warm,
)

# Positions are taken modulo what the tree holds.
POSITIONS = st.integers(min_value=0, max_value=60)
YEARS = st.integers(2014, 2024)
NUMBERS = st.one_of(st.integers(0, 500), st.floats(0, 500, allow_nan=False))

EDITS = st.one_of(
    st.tuples(st.just("update"), POSITIONS, YEARS, NUMBERS),
    st.tuples(st.just("set value"), POSITIONS, POSITIONS, NUMBERS),
    st.tuples(st.just("remove value"), POSITIONS, POSITIONS),
    st.tuples(st.just("delete bracket"), POSITIONS, POSITIONS),
    st.tuples(st.just("append bracket"), POSITIONS, NUMBERS, NUMBERS),
    st.tuples(st.just("add child"), POSITIONS, NUMBERS),
    st.tuples(st.just("remove child"), POSITIONS, POSITIONS),
    st.tuples(
        st.just("set key"),
        POSITIONS,
        st.sampled_from(["label", "unit", "added"]),
        st.text(max_size=3),
    ),
    st.tuples(st.just("read"), POSITIONS),
)
EDIT_LISTS = st.lists(EDITS, max_size=4)
ENDINGS = st.sampled_from(["same tree", "deep copy", "clone", "no tree"])

MODIFY_STEPS = st.one_of(
    st.tuples(st.just("modify"), EDIT_LISTS, ENDINGS),
    st.tuples(st.just("dict"), YEARS, NUMBERS),
)
STEPS = st.lists(
    st.one_of(
        MODIFY_STEPS,
        st.tuples(st.just("direct"), EDITS),
        st.tuples(st.just("read"), POSITIONS),
    ),
    max_size=3,
)
PATHS = st.sampled_from([run_as_instance, run_in_place])
WARMED = st.lists(st.sampled_from(INSTANTS), unique=True).map(tuple)

_SETTINGS = dict(max_examples=200, deadline=None, derandomize=True)


def without_reads(steps):
    """The same steps, with nothing in them that can fill a cache."""
    kept = []
    for step in steps:
        if step[0] == "modify":
            edits = [edit for edit in step[1] if edit[0] != "read"]
            kept.append(("modify", edits, step[2]))
        elif step[0] == "direct":
            if step[1][0] != "read":
                kept.append(step)
        elif step[0] != "read":
            kept.append(step)
    return kept


def never_read(steps):
    """A system given ``steps`` with no cache ever filled: the reference."""
    return run_in_place(new_system(), without_reads(steps))


def warmed_system(instants):
    system = new_system()
    warm(system, instants)
    return system


@settings(**_SETTINGS)
@given(instants=WARMED, steps=STEPS, last=MODIFY_STEPS, run=PATHS)
def test_reads_after_modify_parameters_match_a_tree_never_read(
    instants, steps, last, run
):
    steps = [*steps, last]

    reformed = run(warmed_system(instants), steps)

    assert system_reads(reformed) == system_reads(never_read(steps))


@settings(**_SETTINGS)
@given(instants=WARMED, steps=STEPS, edits=EDIT_LISTS)
def test_a_modifier_that_raises_leaves_a_tree_that_reads_fresh(instants, steps, edits):
    system = run_in_place(warmed_system(instants), steps)

    with pytest.raises(ModifierFailed):
        system.modify_parameters(modifier(edits, "raises"))

    reference = never_read([*steps, ("modify", edits, "same tree")])
    assert system_reads(system) == system_reads(reference)


@settings(**_SETTINGS)
@given(instants=WARMED, steps=STEPS)
def test_the_two_paths_leave_the_same_tree(instants, steps):
    instance = run_as_instance(warmed_system(instants), steps)
    in_place = run_in_place(warmed_system(instants), steps)

    assert snapshot(instance.parameters) == snapshot(in_place.parameters)


@settings(**_SETTINGS)
@given(instants=WARMED, steps=STEPS, raises=st.booleans())
def test_a_reform_instance_changes_nothing_in_its_baseline(instants, steps, raises):
    system = warmed_system(instants)
    before = snapshot(system.parameters)
    cached = cached_nodes(system.parameters)

    if raises:
        steps = [*steps, ("modify", [("set value", 0, 0, 1)], "raises")]
        with pytest.raises(ModifierFailed):
            run_as_instance(system, steps)
    else:
        run_as_instance(system, steps)

    assert snapshot(system.parameters) == before
    assert_keeps_its_cached_nodes(cached)
    assert reads(system.parameters) == reads(new_system().parameters)


@settings(**_SETTINGS)
@given(instants=WARMED, first=EDIT_LISTS, second=EDIT_LISTS, run=PATHS)
def test_two_modifiers_make_the_edits_of_one(instants, first, second, run):
    apart = run(
        warmed_system(instants),
        [("modify", first, "same tree"), ("modify", second, "same tree")],
    )
    together = run(warmed_system(instants), [("modify", first + second, "same tree")])

    assert snapshot(apart.parameters) == snapshot(together.parameters)
    assert system_reads(apart) == system_reads(together)


class DeepCopyingReform(Reform):
    """A reform with ``modify_parameters`` as Core had it before #618."""

    def modify_parameters(self, modifier_function):
        baseline_parameters_copy = copy.deepcopy(self.baseline.parameters)
        reform_parameters = modifier_function(baseline_parameters_copy)
        if not isinstance(reform_parameters, ParameterNode):
            return ValueError("modifier_function must return a ParameterNode")
        self.parameters = reform_parameters
        self._parameters_at_instant_cache = {}


@settings(**_SETTINGS)
@given(edits=EDIT_LISTS, ending=st.sampled_from(["same tree", "deep copy", "clone"]))
def test_one_modifier_matches_the_deep_copy_on_a_baseline_never_read(edits, ending):
    edits = [edit for edit in edits if edit[0] != "read"]
    steps = [("modify", edits, ending)]

    class deep_copying(DeepCopyingReform):
        def apply(self):
            apply_steps(self, steps)

    reference = deep_copying(new_system())
    reformed = run_as_instance(new_system(), steps)

    assert snapshot(reformed.parameters) == snapshot(reference.parameters)
    assert system_reads(reformed) == system_reads(reference)
