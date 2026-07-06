class ParameterKeyWarning(UserWarning):
    """
    Warning raised when a parameter YAML node carries an unknown key beside
    ``values:`` (or beside a scale bracket's components).

    The most damaging case is an ``uprating:`` block written as a sibling of
    ``values:`` instead of nested under ``metadata:``: the file looks like it
    uprates, loads cleanly, and passes CI, while the parameter silently freezes
    at its last explicit value. See issue #505.

    This is emitted as a deprecation-style warning in minor releases; it is
    intended to become a hard error in a future major release once country
    packages have swept their parameter trees.
    """

    pass
