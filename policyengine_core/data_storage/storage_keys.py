"""What a storage key can hold.

Both storages key a value by its branch name and its period's string form:
``f"{branch_name}:{period}"`` in memory, ``f"{branch_name}_{period}"`` on disk
(the name of the value's file). ``check_storable`` enforces the same branch
and period serialization restrictions in both storages (``Holder._set``
chooses between them by how much memory is in use).
"""

from policyengine_core import periods
from policyengine_core.periods import Period


def check_storable(branch_name: str, period: Period) -> None:
    """Raise ``ValueError`` for an ambiguous branch or period serialization.

    Reject a branch name containing ``:``, which separates the parts of an
    in-memory key, and a year or month anchored mid-month, whose string form
    drops the day (``month:2025-03-15`` is ``2025-03``), so that its key would
    be the key of another period. Reject unparseable period strings too,
    including years below 1000 whose string form loses leading zeroes.
    Supported aliases, such as twelve months serialized as one year, retain
    their existing meaning. See policyengine-core#526 for the structured-key
    refactor.
    """
    if ":" in branch_name:
        raise ValueError(
            f"Branch name {branch_name!r} may not contain ':' "
            "(see policyengine-core#526)."
        )
    if period.unit in (periods.YEAR, periods.MONTH) and period.start.day != 1:
        raise ValueError(
            f"Cannot cache period {period} anchored mid-month (it starts on "
            f"{period.start}): its string form is lossy (see "
            "policyengine-core#526)."
        )
    try:
        periods.period(str(period))
    except ValueError as error:
        raise ValueError(
            f"Cannot cache period {period!r}: its string form cannot be parsed "
            "(see policyengine-core#526)."
        ) from error
