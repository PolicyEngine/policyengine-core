from typing import Annotated, Callable, Literal, Any
from datetime import datetime
from dataclasses import dataclass
from policyengine_core.variables import Variable
from policyengine_core.periods import config
from policyengine_core.taxbenefitsystems import TaxBenefitSystem
from policyengine_core.errors import (
    VariableNotFoundError,
    VariableNameConflictError,
)


@dataclass
class TransformationLogItem:
    """
    A log item for a transformation applied to a variable.
    """

    variable_name: str
    variable: Variable | None # None is present when using neutralize
    transformation: Literal["neutralize", "add", "update"]


class StructuralReform:  # Should this inherit from Reform and/or TaxBenefitSystem?

    DEFAULT_START_INSTANT = "0000-01-01"
    transformation_log: list[TransformationLogItem] = []

    tax_benefit_system: TaxBenefitSystem | None = None
    start_instant: Annotated[str, "YYYY-MM-DD"] = DEFAULT_START_INSTANT
    end_instant: Annotated[str, "YYYY-MM-DD"] | None = None

    def add_tax_benefit_system(self, tax_benefit_system: TaxBenefitSystem):
        """
        Add a tax benefit system to the structural reform.

        Args:
          tax_benefit_system: The tax benefit system to be added
        """
        if not isinstance(tax_benefit_system, TaxBenefitSystem):
            raise TypeError(
                "Tax benefit system must be an instance of the TaxBenefitSystem class."
            )
        self.tax_benefit_system = tax_benefit_system

    def activate(self, start_instant: Annotated[str, "YYYY-MM-DD"], end_instant: Annotated[str, "YYYY-MM-DD"] | None):
        """
        Activate the structural reform.

        Args:
          start_instant: The start instant to be added; must be in the format 'YYYY-MM-DD'
          end_instant: The end instant to be added; must be in the format 'YYYY-MM-DD' or None
        """
        if not self.tax_benefit_system or self.tax_benefit_system is None:
            raise ValueError(
                "Tax benefit system must be added before start instant."
            )
        
        self._add_start_instant(start_instant)
        self._add_end_instant(end_instant)
        self._activate_transformation_log()

    def neutralize_variable(self, name: str):
        """
        Neutralize a variable by setting its formula to return the default value
        from the StructuralReform's start_instant date to its end_instant date.

        Args:
          name: The name of the variable
        """
        self.transformation_log.append(
            TransformationLogItem(
                variable_name=name,
                transformation="neutralize",
            )
        )

    def add_variable(self, variable: Variable):
        """
        Add a variable to the StructuralReform.

        Args:
          variable: The variable to be added
        """
        self.transformation_log.append(
            TransformationLogItem(
                variable_name=variable.__name__,
                variable=variable,
                transformation="add",
            )
        )

    def update_variable(self, variable: Variable):
        """
        Update a variable in the tax benefit system; if the variable does not
        yet exist, it will be added.

        Args:
          variable: The variable to be updated
        """
        self.transformation_log.append(
            TransformationLogItem(
                variable_name=variable.__name__,
                variable=variable,
                transformation="update",
            )
        )

    def _activate_transformation_log(self): 
        """
        Activate the transformation log by applying the transformations to the tax benefit system.
        """
        for log_item in self.transformation_log:
            if log_item.transformation == "neutralize":
                self._neutralize_variable(log_item.variable_name)
            elif log_item.transformation == "add":
                self._add_variable(log_item.variable)
            elif log_item.transformation == "update":
                self._update_variable(log_item.variable)

    def _neutralize_variable(self, name: str) -> Variable:
        """
        Neutralize a variable by setting its formula to return the default value
        from the StructuralReform's start_instant date onward.

        Args:
          name: The name of the variable

        Raises:
          VariableNotFoundError: If the variable is not found in the tax benefit system
        """
        # Fetch variable
        fetched_variable: Variable | None = self._fetch_variable(name)

        if fetched_variable is None:
            raise VariableNotFoundError(
                f"Unable to neutralize {name}; variable not found.",
                self.tax_benefit_system,
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

    def _add_variable(self, variable: Variable) -> Variable:
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

        # Insert variable into the tax-benefit system; this will apply default formula over
        # entire period, which we will modify below
        added_variable = self.tax_benefit_system.add_variable(variable)

        # First, neutralize entire period
        neutralized_formula = self._neutralized_formula(variable)
        self._add_formula(
            added_variable,
            neutralized_formula,
            self.DEFAULT_START_INSTANT,
        )

        # Then, re-add formula in order to format correctly
        self._add_formula(
            added_variable,
            variable.formula,
            self.start_instant,
            self.end_instant,
        )

        return added_variable

    def _update_variable(self, variable: Variable) -> Variable:
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

        # If variable doesn't exist, run self._add_variable
        if fetched_variable is None:
            return self._add_variable(variable)

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
        end_instant: Annotated[str, "YYYY-MM-DD"] | None = None,
    ) -> Variable:
        """
        Mutatively add an evolved formula, beginning at start_instant and ending at end_instant,
        to a variable, and return said variable.

        For more on evolved formulas, consult
        https://openfisca.org/doc/coding-the-legislation/40_legislation_evolutions.html

        Args:
          variable: The variable to which the formula will be added
          formula: The formula to be added
          start_instant: The date on which the formula will take effect
          end_instant: The date on which the formula ends, exclusive (i.e.,
            the formula will be applied up to but not including this date); if None,
            the formula will be applied indefinitely

        Returns:
          The variable with the evolved formula
        """
        # Prior to manipulation, get formula at end_instant + 1 day
        next_formula: Callable | None = None
        if end_instant is not None:
            next_formula = variable.get_formula(end_instant)

        # Insert formula into variable's formulas at start_instant
        variable.formulas.update({start_instant: formula})

        # Remove all formulas between start_instant and end_instant (or into perpetuity
        # if end_instant is None)
        for date in variable.formulas.keys():
            if date > start_instant and (
                end_instant is None or date <= end_instant
            ):
                variable.formulas.pop(date)

        # If end_instant, add back in formula at end_instant
        if end_instant is not None:
            variable.formulas[end_instant] = next_formula

        return variable

    def _neutralized_formula(self, variable: Variable) -> Callable:
        """
        Return a formula that itself returns the default value of a variable.

        Args:
          variable: The variable to be neutralized

        Returns:
          The neutralized formula
        """
        return lambda: variable.default_value

    # Validate start instant
    def _validate_instant(self, instant: Any) -> bool:
        """
        Validate an instant.

        Args:
          instant: The instant to be validated
        """
        if not isinstance(instant, str):
            raise TypeError(
                "Instant must be a string in the format 'YYYY-MM-DD'."
            )

        if not config.INSTANT_PATTERN.match(instant):
            raise ValueError(
                "'{}' is not a valid instant. Instants are described using the 'YYYY-MM-DD' format, for instance '2015-06-15'.".format(
                    instant
                )
            )

        return True

    def _add_start_instant(self, start_instant: Annotated[str, "YYYY-MM-DD"]):
        """
        Add a start instant to the structural reform.

        Args:
          start_instant: The start instant to be added
        """

        self._validate_instant(start_instant)
        self.start_instant = start_instant

    def _add_end_instant(self, end_instant: Annotated[str, "YYYY-MM-DD"] | None):
        """
        Add an end instant to the structural reform.

        Args:
          end_instant: The end instant to be added
        """
        if not self.tax_benefit_system or self.tax_benefit_system is None:
            raise ValueError(
                "Tax benefit system must be added before end instant."
            )
        if end_instant is not None:
          self._validate_instant(end_instant)
        self.end_instant = end_instant
    # Default outputs method of some sort?
