"""Systems loaded from the same variable files run the same code objects.

Each system still gets its own modules, classes and functions; only the code
compiled from a file's source is shared. Any change to the source, even one
that keeps the file's size and modification time, is compiled again, and only
the latest version of each file is kept.
"""

import hashlib
import ntpath
import os
import py_compile
import sys
import textwrap
import threading
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


def test_sourceless_bytecode_variable_file_loads_and_runs_formula(tmp_path):
    source = tmp_path / "probe.py"
    bytecode = source.with_suffix(".pyc")
    source.write_text(VARIABLE_FILE.format(value=4))
    try:
        py_compile.compile(str(source), cfile=str(bytecode), doraise=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"Cannot write bytecode: {error}")
    source.unlink()

    system = _system_with_probe(bytecode)
    assert _value(system) == pytest.approx(4)


def test_a_relative_path_compiles_under_its_absolute_path(tmp_path, monkeypatch):
    (tmp_path / "relative_probe.py").write_text(VARIABLE_FILE.format(value=4))
    monkeypatch.chdir(tmp_path)
    by_relative = _system_with_probe("relative_probe.py")
    by_absolute = _system_with_probe(tmp_path / "relative_probe.py")
    code = _formula(by_relative, "shared_code_probe").__code__

    assert code.co_filename == str(tmp_path / "relative_probe.py")
    assert _formula(by_absolute, "shared_code_probe").__code__ is code


@pytest.mark.parametrize("relative", [False, True])
def test_a_symlink_then_parent_path_reads_the_os_resolved_file(
    tmp_path, monkeypatch, relative
):
    repo = tmp_path / "repo"
    other = tmp_path / "other"
    repo.mkdir()
    (other / "subdir").mkdir(parents=True)
    link = repo / "link"
    try:
        link.symlink_to(other / "subdir", target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"Cannot create directory symlinks: {error}")
    posix_target = other / "probe.py"
    posix_target.write_text(VARIABLE_FILE.format(value=1))
    lexical_target = repo / "probe.py"
    lexical_target.write_text(VARIABLE_FILE.format(value=9))
    linked_path = link / ".." / "probe.py"
    if relative:
        monkeypatch.chdir(tmp_path)
        linked_path = linked_path.relative_to(tmp_path)

    # The oracle is open(), as used by the importer: POSIX follows the link
    # before '..', whereas Windows collapses '..' before following the link.
    source_values = {VARIABLE_FILE.format(value=value): value for value in (1, 9)}
    with open(linked_path) as source_file:
        expected_value = source_values[source_file.read()]
    if os.name == "posix":
        assert expected_value == 1
    elif os.name == "nt":
        assert expected_value == 9
    other_path = lexical_target if expected_value == 1 else posix_target
    other_value = 9 if expected_value == 1 else 1

    first = _system_with_probe(linked_path)
    assert _value(first) == pytest.approx(expected_value)
    with open(linked_path, "w") as source_file:
        source_file.write(VARIABLE_FILE.format(value=2))
    second = _system_with_probe(linked_path)
    assert _value(second) == pytest.approx(2)
    assert _value(first) == pytest.approx(expected_value)
    assert _value(_system_with_probe(other_path)) == pytest.approx(other_value)
    assert _formula(first, "shared_code_probe").__code__ is not (
        _formula(second, "shared_code_probe").__code__
    )
    # Distinct OS-selected files need separate entries even with equal source.
    # On POSIX, collapsing '..' would give these two files the same cache key.
    other_path.write_text(VARIABLE_FILE.format(value=2))
    by_other_path = _system_with_probe(other_path)
    assert _value(by_other_path) == pytest.approx(2)
    assert _formula(by_other_path, "shared_code_probe").__code__ is not (
        _formula(second, "shared_code_probe").__code__
    )


@pytest.mark.parametrize("relative", [False, True])
def test_windows_loader_paths_keep_the_os_selected_source(monkeypatch, relative):
    directory = r"C:\project"
    filename = r"repo\link\..\probe.py"
    if not relative:
        filename = ntpath.join(directory, filename)
    absolute_path = ntpath.join(directory, filename)
    lexical_target = r"C:\project\repo\probe.py"
    assert ntpath.normpath(absolute_path) == lexical_target
    sources = {
        lexical_target: b"value = 9\n",
        r"C:\project\other\probe.py": b"value = 1\n",
    }
    reads = []

    def windows_get_data(path):
        # Model ordinary Windows open() normalization, without a Windows host.
        reads.append(path)
        return sources[ntpath.normpath(ntpath.join(directory, path))]

    loader = SourceFileLoader("probe", filename)
    monkeypatch.setattr(loader, "get_data", windows_get_data)
    monkeypatch.setattr(
        tbs_module, "os", types.SimpleNamespace(path=ntpath, getcwd=lambda: directory)
    )
    monkeypatch.setattr(tbs_module, "_variable_file_code", {})

    code = tbs_module._compiled_variable_file(loader)
    namespace = {}
    exec(code, namespace)
    assert namespace["value"] == 9
    assert code.co_filename == absolute_path
    assert tbs_module._variable_file_code == {
        absolute_path: (hashlib.sha256(sources[lexical_target]).digest(), code)
    }
    assert tbs_module._compiled_variable_file(loader) is code
    assert reads == [filename, filename]


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


def test_concurrent_loads_compile_once_and_share_the_code(tmp_path, monkeypatch):
    path = tmp_path / "probe.py"
    path.write_text(VARIABLE_FILE.format(value=3))
    loader = SourceFileLoader("probe", str(path))
    threads = 8
    barrier = threading.Barrier(threads)
    all_attempted = threading.Event()
    observations_lock = threading.Lock()
    attempted = set()
    compilations = 0
    codes, errors = [None] * threads, []

    def record_attempt():
        with observations_lock:
            attempted.add(threading.get_ident())
            if len(attempted) == threads:
                all_attempted.set()

    real_lock = tbs_module._variable_file_code_lock

    class ObservedLock:
        def __enter__(self):
            # Count followers before they block on the first compiler's lock.
            record_attempt()
            return real_lock.__enter__()

        def __exit__(self, *args):
            return real_lock.__exit__(*args)

    class ObservedCache(dict):
        def get(self, key, default=None):
            cached = super().get(key, default)
            # Without the lock, every thread must capture its miss before the
            # first compiler is allowed to publish, including the last thread.
            record_attempt()
            return cached

    source_to_code = loader.source_to_code

    def compile_when_all_attempted(*args, **kwargs):
        nonlocal compilations
        with observations_lock:
            compilations += 1
        # The timeout only diagnoses a deadlock; overlap is controlled by the
        # event, irrespective of the speed or scheduling of the workers.
        assert all_attempted.wait(10), "not all workers reached the cache"
        return source_to_code(*args, **kwargs)

    monkeypatch.setattr(tbs_module, "_variable_file_code_lock", ObservedLock())
    monkeypatch.setattr(tbs_module, "_variable_file_code", ObservedCache())
    monkeypatch.setattr(loader, "source_to_code", compile_when_all_attempted)

    def load(i):
        try:
            barrier.wait(timeout=10)
            codes[i] = tbs_module._compiled_variable_file(loader)
        except BaseException as error:  # reported below, not lost in a thread
            errors.append(error)

    workers = [threading.Thread(target=load, args=(i,)) for i in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=15)

    assert not any(worker.is_alive() for worker in workers)
    assert not errors
    assert len(attempted) == threads
    assert compilations == 1
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
