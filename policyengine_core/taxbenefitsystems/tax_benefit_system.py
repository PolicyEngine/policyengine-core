from __future__ import annotations

import copy
import glob
import importlib
import inspect
import logging
import os
import yaml
from pathlib import Path
import sys
import traceback
import typing
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)
import importlib

from policyengine_core import commons, periods, variables
from policyengine_core.data_structures.reference import Reference
from policyengine_core.entities import Entity
from policyengine_core.errors import (
    VariableNameConflictError,
    VariableNotFoundError,
)
from policyengine_core.parameters import ParameterNode, ParameterNodeAtInstant
from policyengine_core.parameters.operations.homogenize_parameters import (
    homogenize_parameter_structures,
)
from policyengine_core.parameters.operations.interpolate_parameters import (
    interpolate_parameters,
)
from policyengine_core.parameters.operations.propagate_parameter_metadata import (
    propagate_parameter_metadata,
)
from policyengine_core.parameters.operations.uprate_parameters import (
    uprate_parameters,
)
from policyengine_core.periods import Instant, Period
from policyengine_core.populations import GroupPopulation, Population
from policyengine_core.variables import Variable

log = logging.getLogger(__name__)


class TaxBenefitSystem:
    """
    Represents the legislation.

    It stores parameters (values defined for everyone) and variables (values defined for some given entity e.g. a person).

    Attributes:
        parameters: Directory containing the YAML parameter files.

    Args:
        entities: Entities used by the tax benefit system.

    """

    _base_tax_benefit_system: "TaxBenefitSystem" = None
    _parameters_at_instant_cache: Optional[Dict[Any, Any]] = None
    person_key_plural: str = None
    preprocess_parameters: str = None
    baseline: "TaxBenefitSystem" = None  # Baseline tax-benefit system. Used only by reforms. Note: Reforms can be chained.
    cache_blacklist = None
    decomposition_file_path = None
    variable_module_metadata: dict = None

    # The following properties should be specified by country packages.

    entities: List[Entity] = None
    """The entities of the tax and benefit system."""
    variables_dir: str = None
    """Directory containing the Python files defining the variables of the tax and benefit system."""
    parameters_dir: str = None
    """Directory containing the YAML parameter tree."""
    auto_carry_over_input_variables: bool = False
    """Whether to automatically carry over input variables when calculating a variable for a period different from the period of the input variables."""
    basic_inputs: List[str] = None
    """Short list of basic inputs to get medium accuracy."""
    modelled_policies: str = None
    """A YAML filepath containing metadata describing the modelled policies."""

    def __init__(self, entities: Sequence[Entity] = None) -> None:
        if entities is None:
            entities = self.entities

        if entities is None:
            raise ValueError("TaxBenefitSystems must have entities defined.")

        # TODO: Currently: Don't use a weakref, because they are cleared by Paste (at least) at each call.
        self.parameters: Optional[ParameterNode] = None
        self._parameters_at_instant_cache = {}  # weakref.WeakValueDictionary()
        self.variables: Dict[Any, Any] = {}
        # Tax benefit systems are mutable, so entities (which need to know about our variables) can't be shared among them
        if entities is None or len(entities) == 0:
            raise Exception(
                "A tax and benefit sytem must have at least an entity."
            )
        self.entities = [copy.copy(entity) for entity in entities]
        self.person_entity = [
            entity for entity in self.entities if entity.is_person
        ][0]
        self.group_entities = [
            entity for entity in self.entities if not entity.is_person
        ]
        self.group_entity_keys = [entity.key for entity in self.group_entities]
        for entity in self.entities:
            entity.set_tax_benefit_system(self)

        self.variable_module_metadata = {}

        if self.variables_dir is not None:
            self.add_variables_from_directory(self.variables_dir)

        if self.parameters_dir is not None:
            self.load_parameters(self.parameters_dir)
            self.parameters = homogenize_parameter_structures(
                self.parameters, self.variables
            )
            self.parameters = propagate_parameter_metadata(self.parameters)
            self.parameters = interpolate_parameters(self.parameters)
            self.parameters = uprate_parameters(self.parameters)
            self.parameters = propagate_parameter_metadata(self.parameters)
            self.add_abolition_parameters()

        self.add_modelled_policy_metadata()

    def add_abolition_parameters(self):
        if self.parameters is None or "gov" not in self.parameters.children:
            return
        abolition_folder_data = {
            "metadata": {
                "label": "Abolitions",
            }
        }
        for variable in self.variables.values():
            if variable.is_input_variable() or variable.value_type not in (
                bool,
                float,
                int,
            ):
                continue
            abolition_folder_data[variable.name] = {
                "description": f"Set all values of {variable.label} to zero.",
                "values": {
                    "0000-01-01": False,
                },
                "metadata": {
                    "label": f"Abolish {variable.label}",
                    "unit": "bool",
                },
            }
        if not "abolitions" in self.parameters.gov.children:
            self.parameters.gov.add_child(
                "abolitions",
                ParameterNode(
                    name="gov.abolitions", data=abolition_folder_data
                ),
            )

    @property
    def base_tax_benefit_system(self) -> "TaxBenefitSystem":
        base_tax_benefit_system = self._base_tax_benefit_system
        if base_tax_benefit_system is None:
            baseline = self.baseline
            if baseline is None:
                return self
            self._base_tax_benefit_system = (
                base_tax_benefit_system
            ) = baseline.base_tax_benefit_system
        return base_tax_benefit_system

    def instantiate_entities(self) -> Dict[str, Population]:
        person = self.person_entity
        members = Population(person)
        entities: typing.Dict[Entity.key, Entity] = {person.key: members}

        for entity in self.group_entities:
            entities[entity.key] = GroupPopulation(entity, members)

        return entities

    def load_variable(
        self, variable_class: Type[Variable], update: bool = False
    ) -> Variable:
        name = variable_class.__name__

        # Check if a Variable with the same name is already registered.
        baseline_variable = self.get_variable(name)
        if baseline_variable and not update:
            raise VariableNameConflictError(
                f"You've already defined {name} in {baseline_variable.module_name}. You tried to define it again in {variable_class.module_name}."
            )

        variable = variable_class(baseline_variable=baseline_variable)
        self.variables[variable.name] = variable

        return variable

    def add_variable(self, variable: Variable) -> Variable:
        """Adds an OpenFisca variable to the tax and benefit system.

        Args:
            variable: The variable to add. Must be a subclass of Variable.

        Raises:
            policyengine_core.errors.VariableNameConflictError: if a variable with the same name have previously been added to the tax and benefit system.

        """
        return self.load_variable(variable, update=False)

    def replace_variable(self, variable: str) -> Variable:
        """
        Replaces an existing OpenFisca variable in the tax and benefit system by a new one.

        The new variable must have the same name than the replaced one.

        If no variable with the given name exists in the tax and benefit system, no error will be raised and the new variable will be simply added.

        :param Variable variable: New variable to add. Must be a subclass of Variable.
        """
        name = variable.__name__
        if self.variables.get(name) is not None:
            del self.variables[name]
        self.load_variable(variable, update=False)

    def update_variable(self, variable: str) -> Variable:
        """
        Updates an existing OpenFisca variable in the tax and benefit system.

        All attributes of the updated variable that are not explicitely overridden by the new ``variable`` will stay unchanged.

        The new variable must have the same name than the updated one.

        If no variable with the given name exists in the tax and benefit system, no error will be raised and the new variable will be simply added.

        :param Variable variable: Variable to add. Must be a subclass of Variable.
        """
        return self.load_variable(variable, update=True)

    def add_variables_from_file(self, file_path: str) -> None:
        """
        Adds all OpenFisca variables contained in a given file to the tax and benefit system.
        """
        try:
            file_name = os.path.splitext(os.path.basename(file_path))[0]

            path = Path(file_path)

            # Get the relative location, e.g. policyengine_uk/variables/gov/child_benefit.py -> gov.child_benefit
            try:
                relative_file_path = (
                    str(path.relative_to(self.variables_dir))
                    .replace("/", ".")
                    .replace(".py", "")
                )
            except:
                relative_file_path = ""

            #  As Python remembers loaded modules by name, in order to prevent collisions, we need to make sure that:
            #  - Files with the same name, but located in different directories, have a different module names. Hence the file path hash in the module name.
            #  - The same file, loaded by different tax and benefit systems, has distinct module names. Hence the `id(self)` in the module name.
            module_name = (
                f"{id(self)}_{hash(os.path.abspath(file_path))}_{file_name}"
            )

            try:
                spec = importlib.util.spec_from_file_location(
                    module_name, file_path
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            except NameError as e:
                logging.error(
                    str(e)
                    + ": if this code used to work, this error might be due to a major change in OpenFisca-Core. Checkout the changelog to learn more: <https://github.com/openfisca/openfisca-core/blob/master/CHANGELOG.md>"
                )
                raise
            potential_variables = [
                getattr(module, item)
                for item in module.__dict__
                if not item.startswith("__")
            ]

            metadata = {}
            metadata["label"] = module.__dict__.get(
                "label", relative_file_path
            )
            metadata["description"] = module.__dict__.get("description", None)
            metadata["index"] = module.__dict__.get("index", 0)
            self.variable_module_metadata[relative_file_path] = metadata

            i = 0
            for pot_variable in potential_variables:
                # We only want to get the module classes defined in this module (not imported)
                if (
                    inspect.isclass(pot_variable)
                    and issubclass(pot_variable, Variable)
                    and pot_variable.__module__ == module_name
                ):
                    pot_variable.module_name = relative_file_path
                    pot_variable.index_in_module = i
                    i += 1
                    self.add_variable(pot_variable)
        except Exception:
            log.error(
                'Unable to load OpenFisca variables from file "{}"'.format(
                    file_path
                )
            )
            raise

    def add_variable_metadata_from_folder(self, file_path: str) -> None:
        """
        Adds metadata from a given README.md file to the tax and benefit system.
        """
        try:
            path = Path(file_path)

            # Get the relative location, e.g. policyengine_uk/variables/gov/child_benefit.py -> gov.child_benefit
            try:
                relative_file_path = str(
                    path.parent.relative_to(self.variables_dir)
                ).replace("/", ".")
            except:
                relative_file_path = ""

            metadata = {}

            with open(file_path, "r") as f:
                # Get the header as the label (making sure to remove the leading hash), and the rest as the description
                lines = f.readlines()
                metadata["label"] = lines[0].replace("# ", "").strip()
                metadata["description"] = "".join(lines[1:]).strip()

            self.variable_module_metadata[relative_file_path] = metadata
        except Exception:
            log.error(
                'Unable to load OpenFisca metadata from file "{}"'.format(
                    file_path
                )
            )
            raise

    def add_variables_from_directory(self, directory: str) -> None:
        """
        Recursively explores a directory, and adds all OpenFisca variables found there to the tax and benefit system.
        """
        py_files = glob.glob(os.path.join(directory, "*.py"))
        # Try to get the __init__.py file. The __init__.py file may contain metadata about the directory.
        init_module = os.path.join(directory, "__init__.py")
        if init_module in py_files:
            # Extract the label and description from the __init__.py file
            py_files.remove(init_module)
            self.add_variable_metadata_from_folder(init_module)
        if "README.md" in os.listdir(directory):
            self.add_variable_metadata_from_folder(
                os.path.join(directory, "README.md")
            )
        for py_file in py_files:
            self.add_variables_from_file(py_file)
        subdirectories = glob.glob(os.path.join(directory, "*/"))
        for subdirectory in subdirectories:
            self.add_variables_from_directory(subdirectory)

    def add_variables(self, *variables: List[Type[Variable]]):
        """
        Adds a list of OpenFisca Variables to the `TaxBenefitSystem`.

        See also :any:`add_variable`
        """
        for variable in variables:
            self.add_variable(variable)

    def load_extension(self, extension: str) -> None:
        """
        Loads an extension to the tax and benefit system.

        :param str extension: The extension to load. Can be an absolute path pointing to an extension directory, or the name of an OpenFisca extension installed as a pip package.

        """
        # Load extension from installed pip package
        try:
            package = importlib.import_module(extension)
            extension_directory = package.__path__[0]
        except ImportError:
            message = os.linesep.join(
                [
                    traceback.format_exc(),
                    "Error loading extension: `{}` is neither a directory, nor a package.".format(
                        extension
                    ),
                    "Are you sure it is installed in your environment? If so, look at the stack trace above to determine the origin of this error.",
                    "See more at <https://github.com/openfisca/openfisca-extension-template#installing>.",
                ]
            )
            raise ValueError(message)

        self.add_variables_from_directory(extension_directory)
        param_dir = os.path.join(extension_directory, "parameters")
        if os.path.isdir(param_dir):
            extension_parameters = ParameterNode(directory_path=param_dir)
            self.parameters.merge(extension_parameters)

    def apply_reform(self, reform_path: str) -> "TaxBenefitSystem":
        """Generates a new tax and benefit system applying a reform to the tax and benefit system.

        The current tax and benefit system is **not** mutated.

        Args:
            reform_path: The reform to apply. Must respect the format *installed_package.sub_module.reform*

        Returns:
            TaxBenefitSystem: A reformed tax and benefit system.

        Example:

            >>> self.apply_reform('openfisca_france.reforms.inversion_revenus')

        """
        from policyengine_core.reforms import Reform

        try:
            reform_package, reform_name = reform_path.rsplit(".", 1)
        except ValueError:
            raise ValueError(
                "`{}` does not seem to be a path pointing to a reform. A path looks like `some_country_package.reforms.some_reform.`".format(
                    reform_path
                )
            )
        try:
            reform_module = importlib.import_module(reform_package)
        except ImportError:
            message = os.linesep.join(
                [
                    traceback.format_exc(),
                    "Could not import `{}`.".format(reform_package),
                    "Are you sure of this reform module name? If so, look at the stack trace above to determine the origin of this error.",
                ]
            )
            raise ValueError(message)
        reform = getattr(reform_module, reform_name, None)
        if reform is None:
            raise ValueError(
                "{} has no attribute {}".format(reform_package, reform_name)
            )
        if not issubclass(reform, Reform):
            raise ValueError(
                "`{}` does not seem to be a valid Openfisca reform.".format(
                    reform_path
                )
            )

        return reform(self)

    def get_variable(
        self, variable_name: str, check_existence: bool = False
    ) -> Variable:
        """
        Get a variable from the tax and benefit system.

        :param variable_name: Name of the requested variable.
        :param check_existence: If True, raise an error if the requested variable does not exist.
        """
        variables = self.variables
        found = variables.get(variable_name)
        if not found and check_existence:
            raise VariableNotFoundError(variable_name, self)
        return found

    def neutralize_variable(self, variable_name: str) -> None:
        """
        Neutralizes an OpenFisca variable existing in the tax and benefit system.

        A neutralized variable always returns its default value when computed.

        Trying to set inputs for a neutralized variable has no effect except raising a warning.
        """
        self.variables[variable_name] = variables.get_neutralized_variable(
            self.get_variable(variable_name)
        )

    def annualize_variable(
        self, variable_name: str, period: typing.Optional[Period] = None
    ):
        self.variables[variable_name] = variables.get_annualized_variable(
            self.get_variable(variable_name, period)
        )

    def load_parameters(
        self,
        path_to_yaml_dir: str,
    ) -> None:
        """
        Loads the legislation parameter for a directory containing YAML parameters files.

        :param path_to_yaml_dir: Absolute path towards the YAML parameter directory.

        Example:

        >>> self.load_parameters('/path/to/yaml/parameters/dir')
        """

        parameters = ParameterNode(
            "",
            directory_path=path_to_yaml_dir,
        )

        if self.preprocess_parameters is not None:
            parameters = self.preprocess_parameters(parameters)

        self.parameters = parameters

    def _get_baseline_parameters_at_instant(
        self, instant: Instant
    ) -> ParameterNodeAtInstant:
        baseline = self.baseline
        if baseline is None:
            return self.get_parameters_at_instant(instant)
        return baseline._get_baseline_parameters_at_instant(instant)

    def get_parameters_at_instant(
        self, instant: Instant
    ) -> ParameterNodeAtInstant:
        """
        Get the parameters of the legislation at a given instant

        :param instant: :obj:`str` of the format 'YYYY-MM-DD' or :class:`.Instant` instance.
        :returns: The parameters of the legislation at a given instant.
        :rtype: :class:`.ParameterNodeAtInstant`
        """
        if isinstance(instant, Period):
            instant = instant.start
        elif isinstance(instant, (str, int)):
            instant = periods.instant(instant)
        else:
            assert isinstance(
                instant, Instant
            ), "Expected an Instant (e.g. Instant((2017, 1, 1)) ). Got: {}.".format(
                instant
            )

        parameters_at_instant = self._parameters_at_instant_cache.get(instant)
        if parameters_at_instant is None and self.parameters is not None:
            parameters_at_instant = self.parameters.get_at_instant(
                str(instant)
            )
            self._parameters_at_instant_cache[instant] = parameters_at_instant
        return parameters_at_instant

    def get_package_metadata(self) -> dict:
        """
        Gets metatada relative to the country package the tax and benefit system is built from.

        :returns: Country package metadata
        :rtype: dict

        Example:

        >>> tax_benefit_system.get_package_metadata()
        >>> {
        >>>    'location': '/path/to/dir/containing/package',
        >>>    'name': 'openfisca-france',
        >>>    'repository_url': 'https://github.com/openfisca/openfisca-france',
        >>>    'version': '17.2.0'
        >>>    }
        """
        # Handle reforms
        if self.baseline:
            return self.baseline.get_package_metadata()

        fallback_metadata = {
            "name": self.__class__.__name__,
            "version": "",
            "repository_url": "",
            "location": "",
        }

        module = inspect.getmodule(self)
        if not module.__package__:
            return fallback_metadata
        package_name = module.__package__.split(".")[0]
        try:
            distribution = importlib.metadata.distribution(package_name)
        except importlib.metadata.PackageNotFoundError:
            return fallback_metadata

        location = (
            inspect.getsourcefile(module).split(package_name)[0].rstrip("/")
        )
        try:
            home_page_metadatas = [
                metadata.split(":", 1)[1].strip(" ")
                for metadata in distribution._get_metadata(
                    distribution.PKG_INFO
                )
                if "Home-page" in metadata
            ]
        except:
            home_page_metadatas = None
        repository_url = home_page_metadatas[0] if home_page_metadatas else ""
        try:
            return {
                "name": distribution.key,
                "version": distribution.version,
                "repository_url": repository_url,
                "location": location,
            }
        except:
            return fallback_metadata

    def get_variables(self, entity: Entity = None) -> dict:
        """
        Gets all variables contained in a tax and benefit system.

        :param Entity entity: If set, returns only the variable defined for the given entity.

        :returns: A dictionary, indexed by variable names.
        :rtype: dict

        """
        if not entity:
            return self.variables
        else:
            return {
                variable_name: variable
                for variable_name, variable in self.variables.items()
                # TODO - because entities are copied (see constructor) they can't be compared
                if variable.entity.key == entity.key
            }

    def clone(self) -> "TaxBenefitSystem":
        new = commons.empty_clone(self)
        new_dict = new.__dict__

        for key, value in self.__dict__.items():
            if key not in (
                "parameters",
                "_parameters_at_instant_cache",
                "variables",
            ):
                new_dict[key] = value

        new_dict["parameters"] = self.parameters.clone()
        new_dict["_parameters_at_instant_cache"] = {}
        new_dict["variables"] = {
            variable_name: variable.clone()
            for variable_name, variable in self.variables.items()
        }

        for entity in new_dict["entities"]:
            entity.set_tax_benefit_system(new)
        return new

    def entities_plural(self) -> dict:
        return {entity.plural for entity in self.entities}

    def entities_by_singular(self) -> dict:
        return {entity.key: entity for entity in self.entities}

    def test(self, paths: str, verbose: bool = False) -> None:
        from policyengine_core.tools.test_runner import (
            run_tests,
        )  # Import here to avoid circular dependency.

        run_tests(self, paths, options=dict(verbose=verbose))

    def modify_parameters(
        self, modifier_function: Callable[[ParameterNode], ParameterNode]
    ) -> None:
        """Make modifications on the parameters of the legislation.

        Call this function in `apply()` if the reform asks for legislation parameter modifications.

        Args:
            modifier_function: A function that takes a :obj:`.ParameterNode` and should return an object of the same type.
        """
        reform_parameters = modifier_function(self.parameters)
        if not isinstance(reform_parameters, ParameterNode):
            return ValueError(
                "modifier_function {} in module {} must return a ParameterNode".format(
                    modifier_function.__name__,
                    modifier_function.__module__,
                )
            )
        self.parameters = reform_parameters
        self._parameters_at_instant_cache = {}

    def add_modelled_policy_metadata(self):
        """
        Adds metadata to variables that are modelled by PolicyEngine.
        """
        if self.modelled_policies is None:
            return
        with open(self.modelled_policies, "r") as f:
            modelled_policies = yaml.load(f, Loader=yaml.FullLoader)
        self.modelled_policies = modelled_policies
