import copy
import typing

from policyengine_core import commons
from policyengine_core.errors import ParameterParsingError
from policyengine_core.parameters.config import ALLOWED_PARAM_TYPES
from policyengine_core.parameters.helpers import _validate_parameter


class ParameterAtInstant:
    """
    A value of a parameter at a given instant.
    """

    _allowed_keys = set(["value", "metadata"])

    def __init__(
        self,
        name: str,
        instant_str: str,
        data: dict = None,
        file_path: str = None,
        metadata: dict = None,
    ):
        """
        :param str name: name of the parameter, e.g. "taxes.some_tax.some_param"
        :param str instant_str: Date of the value in the format `YYYY-MM-DD`.
        :param dict data: Data, usually loaded from a YAML file.
        """
        self.name: str = name
        self.instant_str: str = instant_str
        self.file_path: str = file_path
        self.metadata: typing.Dict = {}

        # Accept { 2015-01-01: 4000 }
        if not isinstance(data, dict) and isinstance(
            data, ALLOWED_PARAM_TYPES
        ):
            self.value = data
            return

        if metadata is not None:
            self.metadata.update(metadata)  # Inherit metadata from Parameter
        try:
            self.metadata.update(data.get("metadata", {}))
        except Exception as e:
            raise e
        self.validate(data)
        self.value: float = data["value"]

    def validate(self, data: dict) -> None:
        _validate_parameter(
            self, data, data_type=dict, allowed_keys=self._allowed_keys
        )
        try:
            value = data["value"]
        except KeyError:
            raise ParameterParsingError(
                "Missing 'value' property for {}".format(self.name),
                self.file_path,
            )
        if not isinstance(value, ALLOWED_PARAM_TYPES):
            raise ParameterParsingError(
                "Value in {} has type {}, which is not one of the allowed types ({}): {}".format(
                    self.name, type(value), ALLOWED_PARAM_TYPES, value
                ),
                self.file_path,
            )

    def __eq__(self, other) -> bool:
        return (
            (self.name == other.name)
            and (self.instant_str == other.instant_str)
            and (self.value == other.value)
        )

    def __repr__(self) -> str:
        return "ParameterAtInstant({})".format({self.instant_str: self.value})

    def clone(self) -> "ParameterAtInstant":
        clone = commons.empty_clone(self)
        clone.__dict__ = self.__dict__.copy()
        clone.metadata = copy.deepcopy(self.metadata)
        return clone
