from typing import Annotated, Callable
from datetime import datetime
from policyengine_core.variables import Variable
from policyengine_core.taxbenefitsystems import TaxBenefitSystem
from policyengine_core.errors import VariableNotFoundError


class StructuralReform: # Should this inherit from Reform and/or TaxBenefitSystem?

  variables: list[Variable] = []

  def __init__(
      self,
      tax_benefit_system: TaxBenefitSystem,
      start_instant: Annotated[str, "YYYY-MM-DD"] | None = str(datetime.now().year) + "-01-01",
      end_instant: Annotated[str, "YYYY-MM-DD"] | None = None,
  ):
    # TODO: Validate start_instant and end_instant

    self.tax_benefit_system = tax_benefit_system
    self.start_instant = start_instant
    self.end_instant = end_instant

  def neutralize_variable(self, name: str) -> None:
    """
    Neutralize a variable by setting its formula to return the default value
    from the StructuralReform's start_instant date onward.

    Args:
      name: The name of the variable

    Raises:
      VariableNotFoundError: If the variable is not found in the tax benefit system
    """
    # Clone variable
    variable: Variable | None = self._fetch_variable(name)

    if variable is None:
      raise VariableNotFoundError (f"Unable to neutralize {name}; variable not found.")

    # Add formula to variable that returns all defaults
    neutralized_formula = self._neutralized_formula(variable)
    variable = self._add_evolved_formula(variable, neutralized_formula)

  def _fetch_variable(self, name: str) -> Variable | None:
    """
    Fetch the variable by reference from the tax benefit system.

    Args:
      name: The name of the variable
    """
    return self.tax_benefit_system.get_variable(name)
  
  # Method to modify metadata based on new items?

  # Method to add formula based on date
  def _add_evolved_formula(self, variable: Variable, formula: Callable) -> Variable:
    """
    Add an evolved formula, beginning on the StructuralReform's start_instant date, 
    to a variable, and return said variable.

    For more on evolved formulas, consult 
    https://openfisca.org/doc/coding-the-legislation/40_legislation_evolutions.html

    Args:
      variable: The variable to which the formula will be added
      start_instant: The date on which the formula will take effect
      formula: The formula to be added

    Returns:
      The variable with the evolved formula
    """
    
    # Add formula to variable's "formulas" SortedDict
    variable.formulas[self.start_instant] = formula

    return variable
    # 

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