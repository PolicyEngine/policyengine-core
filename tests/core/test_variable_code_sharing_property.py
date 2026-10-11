"""Property test for the shared variable code cache (see test_variable_code_sharing.py)."""

import hashlib
import tempfile
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st

from policyengine_core.taxbenefitsystems import tax_benefit_system as tbs_module
from tests.core.test_variable_code_sharing import VARIABLE_FILE


# Any sequence of writes and loads of one file: every load returns the code of
# the file's current content under its absolute path, a load with no write
# since the last one returns the same object, and the cache holds only the
# latest version.
@settings(max_examples=50, deadline=None)
@given(
    steps=st.lists(
        st.one_of(st.integers(0, 3).map(lambda v: ("write", v)), st.just(("load", 0))),
        min_size=1,
        max_size=12,
    )
)
def test_any_writes_and_loads_give_the_current_code(steps):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory).resolve() / "probe.py"
        path.write_text(VARIABLE_FILE.format(value=0))
        loader = SourceFileLoader("probe", str(path))
        try:
            previous = None
            for action, value in steps:
                if action == "write":
                    path.write_text(VARIABLE_FILE.format(value=value))
                    previous = None
                    continue
                code = tbs_module._compiled_variable_file(loader)
                source = path.read_bytes()
                assert code == compile(source, str(path), "exec", dont_inherit=True)
                assert code.co_filename == str(path)
                if previous is not None:
                    assert code is previous
                previous = code
                digest, cached = tbs_module._variable_file_code[str(path)]
                assert digest == hashlib.sha256(source).digest() and cached is code
        finally:
            tbs_module._variable_file_code.pop(str(path), None)
