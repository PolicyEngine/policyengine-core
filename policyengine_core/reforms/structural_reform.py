from typing import Annotated, Callable
from datetime import datetime
from policyengine_core.periods import instant, Instant
from policyengine_core.variables import Variable
from policyengine_core.taxbenefitsystems import TaxBenefitSystem
from policyengine_core.errors import (
    VariableNotFoundError,
    VariableNameConflictError,
)


class StructuralReform:  # Should this inherit from Reform and/or TaxBenefitSystem?

    DEFAULT_START_INSTANT = "0000-01-01"
    variables: list[Variable] = []

    def __init__(
        self,
        tax_benefit_system: TaxBenefitSystem,
        start_instant: Annotated[str, "YYYY-MM-DD"] | None = str(
            datetime.now().year
        )
        + "-01-01",
        end_instant: Annotated[str, "YYYY-MM-DD"] | None = None,
    ):
        # TODO: Validate start_instant and end_instant

        self.tax_benefit_system = tax_benefit_system
        self.start_instant = start_instant
        self.end_instant = end_instant

    def neutralize_variable(self, name: str) -> Variable:
        """
        Neutralize a variable by setting its formula to return the default value
        from the StructuralReform's start_instant date onward.

        Args:
          name: The name of the variable

        Raises:
          VariableNotFoundError: If the variable is not found in the tax benefit system
        """
        # Clone variable
        fetched_variable: Variable | None = self._fetch_variable(name)

        if fetched_variable is None:
            raise VariableNotFoundError(
                f"Unable to neutralize {name}; variable not found."
            )

        # Add formula to variable that returns all defaults
        neutralized_formula = self._neutralized_formula(fetched_variable)
        self._add_formula(
            fetched_variable,
            neutralized_formula,
            self.start_instant,
            self.end_instant,
        )
        return fetched_variable

    def add_variable(self, variable: Variable) -> None:
        """
        Only partially implemented; Add a variable to the StructuralReform.

        Args:
          variable: The variable to be added

        Raises:
          VariableNameConflictError: If a variable with the same name already exists in the tax benefit system
        """
        if not issubclass(variable, Variable):
            raise TypeError(
                "Variable must be an instance of the Variable class."
            )

        # Attempt to fetch variable
        fetched_variable: Variable | None = self._fetch_variable(
            variable.__name__
        )

        if fetched_variable is not None:
            raise VariableNameConflictError(
                f"Unable to add {variable.__name__}; variable with the same name already exists."
            )

        # TODO: Likely need to do something to add the variable to the TBS

        # Create variable with neutralized formula until start date, then valid formula
        neutralized_formula = self._neutralized_formula(variable)
        self._add_formula(
            variable,
            neutralized_formula,
            self.DEFAULT_START_INSTANT,
            self.start_instant,
        )
        self._add_formula(
            variable, variable.formula, self.start_instant, self.end_instant
        )

    def update_variable(self, variable: Variable) -> Variable:
        """
        Update a variable in the tax benefit system; if the variable does not
        yet exist, it will be added.

        Args:
          variable: The variable to be updated
        """

        # Ensure variable is a Variable
        if not issubclass(variable, Variable):
            raise TypeError(
                "Variable must be an instance of the Variable class."
            )

        # Fetch variable
        fetched_variable: Variable | None = self._fetch_variable(
            variable.__name__
        )

        # If variable doesn't exist, run self.add_variable
        if fetched_variable is None:
            self.add_variable(variable)
            return

        # Otherwise, add new formula to existing variable
        self._add_formula(
            fetched_variable,
            variable.formula,
            self.start_instant,
            self.end_instant,
        )
        return fetched_variable

    def _fetch_variable(self, name: str) -> Variable | None:
        """
        Fetch the variable by reference from the tax benefit system.

        Args:
          name: The name of the variable
        """
        return self.tax_benefit_system.get_variable(name)

    # Method to modify metadata based on new items?

    # Method to add formula based on date
    def _add_formula(
        self,
        variable: Variable,
        formula: Callable,
        start_instant: Annotated[str, "YYYY-MM-DD"],
        end_instant: Annotated[str, "YYYY-MM-DD"] | None,
    ) -> Variable:
        """
        Add an evolved formula, beginning at start_instant and ending at end_instant,
        to a variable, and return said variable.

        For more on evolved formulas, consult
        https://openfisca.org/doc/coding-the-legislation/40_legislation_evolutions.html

        Args:
          variable: The variable to which the formula will be added
          formula: The formula to be added
          start_instant: The date on which the formula will take effect
          end_instant: The final effective date of the formula; any future formula
            begins the next day; if None, the formula will be applied indefinitely

        Returns:
          The variable with the evolved formula
        """

        # Determine if there's already a formula at our exact start_instant
        if start_instant in variable.formulas.keys():
            # If so, save it in case we need it later
            old_formula = variable.formulas[start_instant]

        # Add formula to variable's formulas
        variable.formulas.update({start_instant: formula})

        # If no end_instant, remove all formulas after start_instant
        if end_instant is None:
            for date in variable.formulas.keys():
                if date > start_instant:
                    variable.formulas.pop(date)
            return

        # Otherwise, only remove formulas within interval
        for date in variable.formulas.keys():
            if date > start_instant and date <= end_instant:
                variable.formulas.pop(date)

        # If there's no formula at end_instant + 1 day,
        # add the old formula back in
        typecast_end_instant: Instant = instant(end_instant)
        next_formula_start: str = str(typecast_end_instant.offset(1, "day"))

        if next_formula_start not in variable.formulas.keys():
            variable.formulas[next_formula_start] = old_formula

        return variable

    def _neutralized_formula(self, variable: Variable) -> Callable:
        """
        Return a formula that itself returns the default value of a variable.

        Args:
          variable: The variable to be neutralized

        Returns:
          The neutralized formula
        """
        return lambda population, period, parameters: variable.default_value

    # Validate start instant

    # Default outputs method of some sort?
