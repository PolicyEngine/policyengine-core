import importlib.metadata
import tomli

try:
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    __version__ = pyproject["project"]["version"]
except Exception as e:
    __version__ = importlib.metadata.version("policyengine_core")
