# Transitional imports to ensure non-breaking changes.
# Could be deprecated in the next major release.
#
# How imports are being used today:
#
#   >>> from policyengine_core.module import symbol
#
# The previous example provokes cyclic dependency problems
# that prevent us from modularizing the different components
# of the library so to make them easier to test and to maintain.
#
# How could them be used after the next major release:
#
#   >>> from policyengine_core import module
#   >>> module.symbol()
#
# And for classes:
#
#   >>> from policyengine_core.module import Symbol
#   >>> Symbol()
#
# See: https://www.python.org/dev/peps/pep-0008/#imports

from policyengine_core.projectors import (  # noqa: F401
    Projector,
    EntityToPersonProjector,
    FirstPersonToEntityProjector,
    UniqueRoleToEntityProjector,
)

from policyengine_core.projectors.helpers import (  # noqa: F401
    projectable,
    get_projector_from_shortcut,
)

from .config import ADD, DIVIDE  # noqa: F401
from .population import Population  # noqa: F401
from .group_population import GroupPopulation  # noqa: F401