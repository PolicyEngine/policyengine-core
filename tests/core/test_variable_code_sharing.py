"""Systems loaded from the same variable files run the same code objects.

Each system still gets its own modules, classes and functions; only the code
compiled from a file's source is shared. Any change to the source, even one
that keeps the file's size and modification time, is compiled again, and only
the latest version of each file is kept.
"""

import hashlib
import os
import sys
import textwrap
import threading
import time
import types
from importlib.machinery import SourceFileLoader

import pytest

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.taxbenefitsystems import tax_benefit_system as tbs_module

VARIABLE_FILE = textwrap.dedent(
    """
    from policyengine_core.country_template.entities import Person
    from policyengine_core.periods import MONTH
    from policyengine_core.variables import Variable


    class shared_code_probe(Variable):
        value_type = float
        entity = Person
        definition_period = MONTH
        label = "Shared code probe"

        def formula(person, period, parameters):
            return person("salary", period) * 0 + {value}
    """
)


def _formula(system, name):
    return system.variables[name].get_formula()


def _system_with_probe(path):
    system = CountryTaxBenefitSystem()
    system.add_variables_from_file(str(path))
    return system


def _value(system):
    from policyengine_core.simulations import SimulationBuilder

    simulation = SimulationBuilder().build_from_entities(
        system, {"persons": {"a": {"salary": {"2024-01": 100}}}}
    )
    return simulation.calculate("shared_code_probe", "2024-01")[0]


def test_systems_share_code_but_not_modules_classes_or_functions():
    first = CountryTaxBenefitSystem()
    second = CountryTaxBenefitSystem()
    first_class = type(first.variables["disposable_income"])
    second_class = type(second.variables["disposable_income"])
    first_formula = _formula(first, "disposable_income")
    second_formula = _formula(second, "disposable_income")

    assert first_class is not second_class
    assert (
        sys.modules[first_class.__module__]
        is not (sys.modules[second_class.__module__])
    )
    assert first_formula is not second_formula
    assert first_formula.__globals__ is not second_formula.__globals__
    assert first_formula.__code__ is second_formula.__code__


def test_shared_code_is_compiled_from_the_current_source():
    CountryTaxBenefitSystem()
    assert tbs_module._variable_file_code
    for path, (digest, code) in tbs_module._variable_file_code.items():
        if not os.path.exists(path):
            continue  # a temporary file from another test
        with open(path, "rb") as f:
            source = f.read()
        if hashlib.sha256(source).digest() != digest:
            continue  # a file another test changed after loading it
        assert code == compile(source, path, "exec", dont_inherit=True), path
        # Code equality ignores the file name, so check it separately.
        assert code.co_filename == path


def test_a_relative_path_compiles_under_its_absolute_path(tmp_path, monkeypatch):
    (tmp_path / "relative_probe.py").write_text(VARIABLE_FILE.format(value=4))
    monkeypatch.chdir(tmp_path)
    by_relative = _system_with_probe("relative_probe.py")
    by_absolute = _system_with_probe(tmp_path / "relative_probe.py")
    code = _formula(by_relative, "shared_code_probe").__code__

    assert code.co_filename == str(tmp_path / "relative_probe.py")
    assert _formula(by_absolute, "shared_code_probe").__code__ is code


def test_a_changed_file_is_compiled_again(tmp_path):
    path = tmp_path / "probe.py"
    path.write_text(VARIABLE_FILE.format(value=1))
    first = _system_with_probe(path)
    second = _system_with_probe(path)
    old = tbs_module._variable_file_code[str(path)][1]
    # The same size and modification time: only the content tells them apart.
    stat = os.stat(path)
    path.write_text(VARIABLE_FILE.format(value=2))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert os.stat(path).st_size == stat.st_size
    third = _system_with_probe(path)

    assert _formula(first, "shared_code_probe").__code__ is (
        _formula(second, "shared_code_probe").__code__
    )
    assert _formula(third, "shared_code_probe").__code__ is not (
        _formula(first, "shared_code_probe").__code__
    )
    assert _value(first) == pytest.approx(1)
    assert _value(second) == pytest.approx(1)
    assert _value(third) == pytest.approx(2)
    # Only the new version is kept.
    digest, _ = tbs_module._variable_file_code[str(path)]
    assert digest == hashlib.sha256(path.read_bytes()).digest()
    assert not any(code is old for _, code in tbs_module._variable_file_code.values())


class _CountingLoader(SourceFileLoader):
    """Counts compilations and sleeps in each, so that threads overlap."""

    compilations = 0

    def source_to_code(self, data, path, *args, **kwargs):
        type(self).compilations += 1
        time.sleep(0.05)  # releases the GIL
        return super().source_to_code(data, path, *args, **kwargs)


def test_concurrent_loads_compile_once_and_share_the_code(tmp_path, monkeypatch):
    path = tmp_path / "probe.py"
    path.write_text(VARIABLE_FILE.format(value=3))
    monkeypatch.setattr(_CountingLoader, "compilations", 0)
    loader = _CountingLoader("probe", str(path))
    threads = 8
    barrier = threading.Barrier(threads)
    codes, errors = [None] * threads, []

    def load(i):
        try:
            barrier.wait()
            codes[i] = tbs_module._compiled_variable_file(loader)
        except BaseException as error:  # reported below, not lost in a thread
            errors.append(error)

    workers = [threading.Thread(target=load, args=(i,)) for i in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert not errors
    assert _CountingLoader.compilations == 1
    assert all(isinstance(code, types.CodeType) for code in codes)
    assert all(code is codes[0] for code in codes)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_forked_child_gets_a_free_lock():
    # A child forked while another thread held the lock would deadlock on its
    # first variable file unless the lock is re-created after fork.
    with tbs_module._variable_file_code_lock:
        pid = os.fork()
        if pid == 0:
            status = 2
            try:
                status = 1 if tbs_module._variable_file_code_lock.locked() else 0
            finally:
                os._exit(status)
        _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
