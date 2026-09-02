class NonDeterministicFormulaError(RuntimeError):
    """A variable's formula references a random number generator.

    Rules-engine formulas must be deterministic functions of their inputs.
    Raised at variable registration by
    ``policyengine_core.variables.formula_randomness.check_formula_determinism``.
    """

    pass
