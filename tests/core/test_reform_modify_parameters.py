"""``modify_parameters`` gives a modifier the system's own tree and reads it fresh.

A reform runs as an instance (``MyReform(baseline)``) or in place
(``system.apply_reform_set(MyReform)``). ``Reform.modify_parameters`` used to
deep-copy the baseline's tree for each call, which had two effects on the
instance path. It threw away what ``apply()`` had changed so far, and it carried
the baseline's cached nodes into the copy. A modifier that edited values,
brackets or children without ``Parameter.update`` then read the baseline's
values at any instant the baseline had cached. In place, the modifier gets the
system's own tree, with the same stale nodes.

Both paths now run ``TaxBenefitSystem.modify_parameters``. See #618.

``test_reform_modify_parameters_property.py`` has the properties.
"""

import copy

import pytest

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.country_template.reforms.modify_social_security_taxation import (
    modify_social_security_taxation,
)
from policyengine_core.parameters import ParameterNode, ParameterScaleBracket
from policyengine_core.periods import instant
from policyengine_core.reforms import Reform, set_parameter
from tests.fixtures.reform_modifiers import (
    ModifierFailed,
    assert_keeps_its_cached_nodes,
    cached_nodes,
    new_system,
    read,
    system_reads,
    warm,
)

INSTANT = "2017-01-01"


def as_instance(reform, system):
    return reform(system)


def in_place(reform, system):
    system.apply_reform_set(reform)
    return system


BOTH_PATHS = pytest.mark.parametrize(
    "run", [as_instance, in_place], ids=["as an instance", "in place"]
)


def social_security_thresholds(system):
    return system.get_parameters_at_instant(
        INSTANT
    ).taxes.social_security_contribution.thresholds


def double_basic_income(parameters):
    parameters.benefits.basic_income.update(
        period=f"year:{INSTANT}:1",
        value=2 * parameters.benefits.basic_income(INSTANT),
    )
    return parameters


# ----- Stale values ----- #


class modify_brackets_in_place(Reform):
    """The template reform, in the form ``apply_reform_set`` can run: there
    ``self`` is the system, which has no ``modify_brackets``."""

    def apply(self):
        self.modify_parameters(modify_social_security_taxation.modify_brackets)


@pytest.mark.parametrize(
    "run, reform",
    [
        (as_instance, modify_social_security_taxation),
        (in_place, modify_brackets_in_place),
    ],
    ids=["as an instance", "in place"],
)
@pytest.mark.parametrize("read_first", [True, False], ids=["read", "never read"])
def test_template_reform_reads_its_modifier(run, reform, read_first):
    system = CountryTaxBenefitSystem()
    if read_first:
        assert social_security_thresholds(system) == [0.0, 6000.0, 12400.0]

    reformed = run(reform, system)

    assert social_security_thresholds(reformed) == [6000.0, 12400.0, 40000]


def assign_a_value(parameters):
    parameters.benefit.amount.values_list[0].value = -1


def remove_a_value(parameters):
    parameters.benefit.amount.values_list.pop(0)


def assign_a_brackets_value(parameters):
    parameters.tax.brackets[1].rate.values_list[0].value = -1


def delete_a_bracket(parameters):
    del parameters.tax.brackets[0]


def append_a_bracket(parameters):
    parameters.tax.brackets.append(
        ParameterScaleBracket(
            "new_bracket",
            data={
                "threshold": {"2015-01-01": {"value": 5_000}},
                "rate": {"2015-01-01": {"value": 0.5}},
            },
        )
    )


def add_a_child(parameters):
    parameters.benefit.add_child("new", ParameterNode(data={"leaf": {"2015-01-01": 1}}))


def remove_a_child(parameters):
    del parameters.benefit.children["eligible"]


def update_a_parameter_in_a_bracket(parameters):
    """A loaded bracket has no ``parent``, so ``update`` stops at the bracket."""
    parameters.tax.brackets[1].rate.update(period="year:2018:1", value=-1)


@BOTH_PATHS
@pytest.mark.parametrize(
    "edit",
    [
        assign_a_value,
        remove_a_value,
        assign_a_brackets_value,
        delete_a_bracket,
        append_a_bracket,
        add_a_child,
        remove_a_child,
        update_a_parameter_in_a_bracket,
    ],
)
def test_edits_of_a_warm_tree_read_as_a_cold_tree_of_the_same_edits(run, edit):
    def modify(parameters):
        edit(parameters)
        return parameters

    def reform(warm_first):
        class edited(Reform):
            def apply(self):
                if warm_first:
                    warm(self)
                self.modify_parameters(modify)

        return edited

    warmed = new_system()
    warm(warmed)
    reformed = run(reform(True), warmed)

    assert system_reads(reformed) == system_reads(run(reform(False), new_system()))


def test_dict_form_clears_what_an_update_under_a_bracket_leaves_behind():
    """A loaded bracket has no ``parent``, so ``update`` stops at the bracket."""
    system = new_system()
    warm(system)

    system.modify_parameters({"tax[1].rate": {"year:2016:1": 0.9}})

    assert system.get_parameters_at_instant("2016-01-01").tax.rates == [0.1, 0.9]
    assert system.parameters("2016-01-01").tax.rates == [0.1, 0.9]


@BOTH_PATHS
@pytest.mark.parametrize(
    "replace", [copy.deepcopy, ParameterNode.clone], ids=["deep copy", "clone"]
)
def test_a_tree_the_modifier_returns_reads_fresh(run, replace):
    def modify(parameters):
        parameters = replace(parameters)
        parameters.benefit.amount.values_list[0].value = -1
        return parameters

    class replaces_the_tree(Reform):
        def apply(self):
            warm(self)
            self.modify_parameters(modify)

    system = new_system()
    warm(system)

    reformed = run(replaces_the_tree, system)

    assert reformed.parameters(INSTANT).benefit.amount == 100
    assert reformed.parameters("2020-01-01").benefit.amount == -1
    assert reformed.get_parameters_at_instant("2020-01-01").benefit.amount == -1


def test_a_tree_the_modifier_edits_and_drops_reads_fresh():
    """Anything else holding that tree sees the edits as well."""
    system = new_system()
    warm(system)
    tree = system.parameters

    def modify(parameters):
        parameters.benefit.amount.values_list[0].value = -1
        return ParameterNode(data={"leaf": {"2015-01-01": 1}})

    system.modify_parameters(modify)

    assert tree("2020-01-01").benefit.amount == -1
    assert system.parameters("2020-01-01").leaf == 1


def test_a_modifier_that_raises_leaves_a_tree_that_reads_fresh():
    """The exception reaches the caller, who may keep using the system."""
    system = new_system()
    warm(system)

    def modify(parameters):
        parameters.benefit.amount.values_list[0].value = -1
        raise ModifierFailed

    with pytest.raises(ModifierFailed):
        system.modify_parameters(modify)

    assert system.get_parameters_at_instant("2020-01-01").benefit.amount == -1
    assert system.parameters("2020-01-01").benefit.amount == -1


@BOTH_PATHS
def test_edits_made_before_a_modifier_raises_stay(run):
    """``apply()`` may catch the exception and carry on."""

    def modify(parameters):
        parameters.benefit.amount.values_list[0].value = -1
        raise ModifierFailed

    class catches_it(Reform):
        def apply(self):
            warm(self)
            with pytest.raises(ModifierFailed):
                self.modify_parameters(modify)

    reformed = run(catches_it, new_system())

    assert reformed.get_parameters_at_instant("2020-01-01").benefit.amount == -1


@BOTH_PATHS
def test_a_read_through_the_system_inside_a_modifier_leaves_nothing_stale(run):
    class reads_then_edits(Reform):
        def apply(self):
            def modify(parameters):
                self.get_parameters_at_instant("2020-01-01")
                parameters.benefit.amount.values_list[0].value = -1
                return parameters

            self.modify_parameters(modify)

    reformed = run(reads_then_edits, new_system())

    assert reformed.get_parameters_at_instant("2020-01-01").benefit.amount == -1


def test_the_system_gets_a_new_cache_and_leaves_the_old_one_alone():
    """policyengine-us's SPM policies share this dict, so emptying it in place
    would reach the others."""
    system = new_system()
    warm(system)
    shared = system._parameters_at_instant_cache
    instants = set(shared)

    system.modify_parameters(lambda parameters: parameters)

    assert instants
    assert system._parameters_at_instant_cache is not shared
    assert set(shared) == instants


def test_clear_descendant_caches_reaches_every_node_below_and_no_other():
    system = new_system()
    parameters = system.parameters
    parameters.benefit.add_child(
        "inner", ParameterNode(data={"node": {"leaf": {"2015-01-01": 1}}})
    )
    warm(system)
    below = [
        parameters.benefit,
        parameters.benefit.inner,
        parameters.benefit.inner.node,
    ]
    nodes = [node for node, _ in cached_nodes(parameters).values()]
    assert all(node._at_instant_cache for node in nodes)

    parameters.benefit.clear_descendant_caches()

    for node in nodes:
        assert bool(node._at_instant_cache) != any(node is item for item in below)

    parameters.clear_descendant_caches()

    assert not any(node._at_instant_cache for node in nodes)


# ----- Redundant copy, and the changes it dropped ----- #


@BOTH_PATHS
def test_changes_made_earlier_in_apply_are_kept(run):
    class update_then_two_modifiers(Reform):
        def apply(self):
            self.parameters.taxes.income_tax_rate.update(
                period=f"year:{INSTANT}:1", value=0.2
            )
            self.modify_parameters(double_basic_income)
            self.modify_parameters(double_basic_income)

    reformed = run(update_then_two_modifiers, CountryTaxBenefitSystem())

    at_instant = reformed.get_parameters_at_instant(INSTANT)
    assert at_instant.taxes.income_tax_rate == 0.2
    assert at_instant.benefits.basic_income == 2400


@BOTH_PATHS
def test_set_parameter_reforms_build_on_each_other(run):
    first = set_parameter(
        "taxes.income_tax_rate", 0.2, period="year:2017:1", return_modifier=True
    )
    second = set_parameter(
        "benefits.basic_income", 900, period="year:2017:1", return_modifier=True
    )

    class set_both(Reform):
        def apply(self):
            self.modify_parameters(first)
            self.modify_parameters(second)

    at_instant = run(set_both, CountryTaxBenefitSystem()).get_parameters_at_instant(
        INSTANT
    )

    assert at_instant.taxes.income_tax_rate == 0.2
    assert at_instant.benefits.basic_income == 900


def test_a_reform_of_a_reform_keeps_the_first_reforms_changes():
    class first(Reform):
        def apply(self):
            self.modify_parameters(double_basic_income)

    class second(Reform):
        def apply(self):
            self.modify_parameters(double_basic_income)

    system = CountryTaxBenefitSystem()
    once = first(system)
    twice = second(once)

    assert system.parameters.benefits.basic_income(INSTANT) == 600
    assert once.parameters.benefits.basic_income(INSTANT) == 1200
    assert twice.parameters.benefits.basic_income(INSTANT) == 2400


def test_a_reform_hands_its_modifier_the_tree_it_already_holds(monkeypatch):
    system = new_system()
    handed = []

    def fail_to_deep_copy(self, memo):
        raise AssertionError("A reform has no reason to deep-copy a parameter tree.")

    monkeypatch.setattr(ParameterNode, "__deepcopy__", fail_to_deep_copy, raising=False)

    class records_its_tree(Reform):
        def apply(self):
            tree = self.parameters

            def modify(parameters):
                handed.append(parameters is tree)
                return parameters

            self.modify_parameters(modify)

    assert records_its_tree(system).parameters is not system.parameters
    assert handed == [True]


@BOTH_PATHS
def test_a_modifier_that_returns_no_tree_keeps_its_edits(run):
    """It still gets ``ValueError`` back rather than raised: see #618."""
    results = []

    def modify(parameters):
        parameters.benefit.amount.values_list[0].value = -1

    class returns_nothing(Reform):
        def apply(self):
            warm(self)
            results.append(self.modify_parameters(modify))

    system = new_system()
    warm(system)

    reformed = run(returns_nothing, system)

    assert type(results[0]) is ValueError
    assert reformed.get_parameters_at_instant("2020-01-01").benefit.amount == -1


@BOTH_PATHS
@pytest.mark.parametrize(
    "change",
    [lambda parameters: parameters, {"benefit.amount": {"year:2020:1": -1}}],
    ids=["function", "dict"],
)
def test_modify_parameters_returns_the_system_it_ran_on(run, change):
    results = []

    class records(Reform):
        def apply(self):
            results.append((self.modify_parameters(change), self))

    run(records, new_system())

    assert results[0][0] is results[0][1]


def test_a_reform_instance_takes_the_dict_form():
    class dict_form(Reform):
        def apply(self):
            self.modify_parameters({"benefit.amount": {"year:2020:1": -1}})

    system = new_system()
    warm(system)

    assert dict_form(system).parameters("2020-01-01").benefit.amount == -1
    assert system.parameters("2020-01-01").benefit.amount == 140


# ----- The baseline ----- #


def test_a_reform_leaves_its_baseline_and_the_baselines_caches_alone():
    system = new_system()
    warm(system)
    expected = read(system.parameters, INSTANT)
    cached = cached_nodes(system.parameters)
    at_instant = system.get_parameters_at_instant(INSTANT)

    class edits_the_tree(Reform):
        def apply(self):
            self.modify_parameters(modify_many_ways)

    def modify_many_ways(parameters):
        parameters.benefit.amount.update(period=f"year:{INSTANT}:1", value=-1)
        parameters.benefit.amount.values_list[-1].value = -2
        parameters.tax.brackets[1].rate.values_list[0].value = -3
        parameters.tax.brackets.pop(0)
        parameters.allowance.children.pop("single")
        parameters.benefit.add_child(
            "new", ParameterNode(data={"leaf": {"2015-01-01": 1}})
        )
        return parameters

    edits_the_tree(system)

    assert_keeps_its_cached_nodes(cached)
    assert system.get_parameters_at_instant(INSTANT) is at_instant
    assert read(system.parameters.clone(), INSTANT) == expected


def test_replacing_metadata_and_list_values_stays_out_of_the_baseline():
    """A clone shares nested metadata and list values with what it was cloned
    from, so a modifier replaces them. Editing one in place, which the deep copy
    allowed, would reach the baseline: see ``modify_parameters``."""
    system = new_system()
    system.parameters.add_child(
        "sources", ParameterNode(data={"list": {"2015-01-01": ["salary"]}})
    )

    def modify(parameters):
        amount = parameters.benefit.amount
        amount.metadata["reference"] = [*amount.metadata["reference"], "new"]
        amount.metadata["new"] = True
        entry = parameters.sources.list.values_list[0]
        entry.value = [*entry.value, "pension"]
        return parameters

    class replaces(Reform):
        def apply(self):
            self.modify_parameters(modify)

    reformed = replaces(system)

    assert reformed.parameters.sources.list(INSTANT) == ["salary", "pension"]
    assert len(reformed.parameters.benefit.amount.metadata["reference"]) == 2
    assert system.parameters.sources.list(INSTANT) == ["salary"]
    assert len(system.parameters.benefit.amount.metadata["reference"]) == 1
    assert "new" not in system.parameters.benefit.amount.metadata


def test_a_reform_of_a_system_with_no_parameters_can_supply_a_tree():
    system = new_system()
    system.parameters = None

    class supplies_a_tree(Reform):
        def apply(self):
            self.modify_parameters(
                lambda parameters: ParameterNode(data={"leaf": {"2015-01-01": 1}})
            )

    reformed = supplies_a_tree(system)

    assert reformed.get_parameters_at_instant(instant("2016-01-01")).leaf == 1
