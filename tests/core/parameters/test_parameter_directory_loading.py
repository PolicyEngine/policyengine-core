"""Loading a parameter tree from directories that also hold non-parameter files.

Country packages keep some lookup tables as CSV files beside an ``__init__.py``
that variables import, so Python writes ``__pycache__`` into those parameter
directories after the first import. The loaded tree must not depend on whether
that has happened.
"""

import py_compile
import sys

from policyengine_core.parameters import ParameterNode


def _write_parameter_tree(root):
    """Write one YAML parameter and one package directory holding only a CSV."""
    benefit = root / "benefit"
    benefit.mkdir()
    (benefit / "amount.yaml").write_text("values:\n  2020-01-01: 100\n")
    lookup = root / "lookup"
    lookup.mkdir()
    (lookup / "__init__.py").write_text('RATES_FILE = "rates.csv"\n')
    (lookup / "rates.csv").write_text("zip_code,rate\n00001,1.0\n")


def _write_bytecode_cache(package_directory):
    """Write the ``__pycache__`` entry that importing the package leaves."""
    source = package_directory / "__init__.py"
    cache_name = f"__init__.{sys.implementation.cache_tag}.pyc"
    cache = package_directory / "__pycache__" / cache_name
    py_compile.compile(str(source), cfile=str(cache), doraise=True)


def _parameter_node_names(directory):
    tree = ParameterNode("", directory_path=str(directory))
    return sorted(
        descendant.name
        for descendant in tree.get_descendants()
        if isinstance(descendant, ParameterNode)
    )


def test_bytecode_cache_does_not_change_parameter_nodes(tmp_path):
    _write_parameter_tree(tmp_path)
    node_names_without_cache = _parameter_node_names(tmp_path)

    _write_bytecode_cache(tmp_path / "lookup")
    assert (tmp_path / "lookup" / "__pycache__").is_dir()
    node_names_with_cache = _parameter_node_names(tmp_path)

    assert node_names_without_cache == ["benefit", "lookup"]
    assert node_names_with_cache == node_names_without_cache
    tree = ParameterNode("", directory_path=str(tmp_path))
    assert tree.children["lookup"].children == {}


def test_hidden_directories_are_not_loaded(tmp_path):
    _write_parameter_tree(tmp_path)
    checkpoints = tmp_path / "benefit" / ".ipynb_checkpoints"
    checkpoints.mkdir()
    (checkpoints / "amount-checkpoint.yaml").write_text("values:\n  2020-01-01: 100\n")

    tree = ParameterNode("", directory_path=str(tmp_path))

    assert sorted(tree.children["benefit"].children) == ["amount"]
