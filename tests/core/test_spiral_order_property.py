"""Property: recursions over periods give results independent of order.

For any system of one or two yearly variables, each equal to 1 plus other
variables (itself included) at lags of 0 to 2 years, with dated or undated
formulas, inputs at random years and any ``max_spiral_loops``, every request
returns the same value, or raises the same error, after any earlier requests
as it does in a new simulation.
"""

import pytest

from tests.fixtures.spirals import build, make_recurrence, run

hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

NAMES = ["r0", "r1"]
YEARS = st.integers(2000, 2035)


@st.composite
def systems(draw):
    names = NAMES[: draw(st.integers(1, 2))]
    variables = []
    for name in names:
        terms = draw(
            st.lists(
                st.tuples(st.sampled_from(names), st.integers(0, 2)),
                min_size=1,
                max_size=2,
            )
        )
        # Lag 0 on itself is a plain cycle, tested elsewhere.
        terms = [(other, lag or (1 if other == name else 0)) for other, lag in terms]
        start = draw(st.sampled_from([None, 2004, 2008]))
        variables.append((name, tuple(terms), start))
    inputs = draw(
        st.dictionaries(
            st.tuples(st.sampled_from(names), st.integers(2000, 2012)),
            st.integers(1, 9),
            max_size=2,
        )
    )
    loops = draw(st.sampled_from([1, 2, 3, 10]))
    requests = st.tuples(st.sampled_from(names), YEARS)
    prior = draw(st.lists(requests, max_size=4))
    target = draw(requests)
    return variables, inputs, loops, prior, target


@hypothesis.settings(max_examples=200, deadline=None)
@hypothesis.given(systems())
def test_earlier_requests_do_not_change_a_result(system):
    variables, inputs, loops, prior, target = system

    def new():
        return build(
            [make_recurrence(name, terms, start) for name, terms, start in variables],
            inputs,
            loops,
        )

    used = new()
    for request in prior:
        run(used, *request)
    assert run(used, *target) == run(new(), *target)
