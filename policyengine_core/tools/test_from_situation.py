import yaml
from pathlib import Path
import numpy as np
import json


def generate_test_from_situation(situation: dict, file_path: str):
    """Generate a test from a situation.

    Args:
        situation (dict): The situation to generate the test from.
        test_name (str): The name of the test.
    """

    yaml_contents = [
        {
            "input": situation,
            "output": {},
        }
    ]

    with open(Path(file_path), "w+") as f:
        yaml.dump(yaml_contents, f)
