"""Run batches of YAML tests in processes forked from one that built the system.

Starting a process for each batch of a large country suite repeats the same
work every time: policyengine-us takes about 40 seconds to import and build
its tax-benefit system before the first case runs, and over half a minute to
tear its interpreter down afterwards. Forking from a process that has already
built the system skips both, and the children share the parent's copy of the
system in memory until they write to it, so several batches can run at once
in the memory one used to take.
"""

import gc
import os
import signal
import sys
import tempfile
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from policyengine_core.tools.test_runner import _get_tax_benefit_system, run_tests

# pytest's exit code for an internal error, used when a child fails outside
# pytest (for example, opening its log).
PYTEST_INTERNAL_ERROR = 3
# How long a child gets to exit after SIGTERM before it is killed.
TERMINATE_GRACE_SECONDS = 5
# How often the parent checks on its children.
POLL_SECONDS = 0.1


@dataclass
class BatchResult:
    """How one batch ended.

    ``exit_code`` is pytest's exit code, negative for a child ended by a
    signal (as ``subprocess`` reports it). ``peak_rss_mb`` is the child's
    peak resident set, which counts the pages it still shares with the parent.
    """

    paths: List[str]
    exit_code: int
    elapsed: float
    peak_rss_mb: Optional[float]
    output: str
    timed_out: bool = False


def run_batches_in_forks(
    tax_benefit_system,
    batches: Sequence[Sequence[str]],
    options: Optional[dict] = None,
    workers: int = 1,
    timeout: Optional[float] = None,
    on_finish: Optional[Callable[[int, BatchResult], None]] = None,
    on_heartbeat: Optional[Callable[[Dict[int, float]], None]] = None,
    heartbeat_seconds: float = 60,
) -> List[BatchResult]:
    """Run each batch of test paths in its own child process, ``workers`` at once.

    Each child is forked from this process, runs ``run_tests`` on its batch
    with ``tax_benefit_system`` and ``options``, and exits as soon as pytest
    returns. Its output goes to a file and is returned whole in its result,
    so concurrent batches never interleave. Batches start in the order given.

    ``timeout`` bounds each batch in seconds: a batch still running then is
    sent SIGTERM, then SIGKILL, and reported with ``timed_out``.
    ``on_finish(index, result)`` is called as each batch ends, and
    ``on_heartbeat({index: seconds running})`` every ``heartbeat_seconds``
    while any runs.

    Call this from a process with no other threads: a thread holding a lock
    when the process forks leaves that lock held forever in the child. That
    includes a numerical library's thread pool, so set
    ``OPENBLAS_NUM_THREADS=1`` (and ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS``)
    before the process imports numpy.

    Before forking, this builds the reform-free system the runner gives every
    case without reforms (see ``_get_tax_benefit_system``), so the children
    share it rather than each building their own, and moves every object the
    parent holds to the garbage collector's permanent generation
    (``gc.freeze``), so a child's collections do not write to, and copy, the
    pages it shares with the parent.
    """
    if not hasattr(os, "fork"):
        raise NotImplementedError("Running batches in forks needs os.fork.")
    if workers < 1:
        raise ValueError(f"workers must be 1 or more, got {workers}")
    if options is None:
        options = {}

    _get_tax_benefit_system(tax_benefit_system, [], [])

    pending = deque(enumerate(batches))
    running: Dict[int, tuple] = {}
    results: List[Optional[BatchResult]] = [None] * len(batches)
    last_heartbeat = time.monotonic()
    log_directory = tempfile.mkdtemp(prefix="policyengine-batches-")
    gc.collect()
    gc.freeze()
    try:
        while pending or running:
            while pending and len(running) < workers:
                index, paths = pending.popleft()
                log_path = os.path.join(log_directory, f"batch-{index}.log")
                sys.stdout.flush()
                sys.stderr.flush()
                pid = os.fork()
                if pid == 0:
                    _run_child(tax_benefit_system, list(paths), options, log_path)
                running[pid] = (index, list(paths), log_path, time.monotonic(), None)

            any_finished = False
            for pid, (index, paths, log_path, started, terminated_at) in list(
                running.items()
            ):
                waited_pid, status, usage = os.wait4(pid, os.WNOHANG)
                now = time.monotonic()
                if waited_pid == 0:
                    if terminated_at is None:
                        if timeout is not None and now - started >= timeout:
                            os.kill(pid, signal.SIGTERM)
                            running[pid] = (index, paths, log_path, started, now)
                    elif now - terminated_at >= TERMINATE_GRACE_SECONDS:
                        os.kill(pid, signal.SIGKILL)
                    continue
                any_finished = True
                del running[pid]
                result = BatchResult(
                    paths=paths,
                    exit_code=os.waitstatus_to_exitcode(status),
                    elapsed=now - started,
                    peak_rss_mb=_peak_rss_mb(usage),
                    output=_read_and_remove(log_path),
                    timed_out=terminated_at is not None,
                )
                results[index] = result
                if on_finish is not None:
                    on_finish(index, result)

            now = time.monotonic()
            if (
                on_heartbeat is not None
                and running
                and (now - last_heartbeat >= heartbeat_seconds)
            ):
                on_heartbeat(
                    {
                        index: now - started
                        for index, _, _, started, _ in running.values()
                    }
                )
                last_heartbeat = now
            if not any_finished:
                time.sleep(POLL_SECONDS)
    finally:
        # Reached with children still running only if the parent was
        # interrupted: do not leave them behind.
        for pid in running:
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except (ChildProcessError, ProcessLookupError):
                pass
        gc.unfreeze()
        for name in os.listdir(log_directory):
            os.remove(os.path.join(log_directory, name))
        os.rmdir(log_directory)
    return results


def _run_child(tax_benefit_system, paths, options, log_path) -> None:
    """Run one batch in a forked child, then exit without returning."""
    exit_code = PYTEST_INTERNAL_ERROR
    try:
        log = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        # The standard descriptors, and whatever ``sys.stdout`` and
        # ``sys.stderr`` write to if something (pytest's capture) replaced them.
        for descriptor in {1, 2, *_stream_descriptors()}:
            os.dup2(log, descriptor)
        os.close(log)
        exit_code = int(run_tests(tax_benefit_system, paths, options))
    except BaseException:
        traceback.print_exc()
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            # Skip interpreter teardown: freeing a large country's systems
            # one object at a time took over half a minute, and nothing the
            # child holds needs releasing before the process goes.
            os._exit(exit_code)


def _stream_descriptors() -> List[int]:
    descriptors = []
    for stream in (sys.stdout, sys.stderr):
        try:
            descriptors.append(stream.fileno())
        except (AttributeError, OSError, ValueError):
            pass
    return descriptors


def _peak_rss_mb(usage) -> Optional[float]:
    """A finished child's peak resident set in MB, from ``os.wait4``."""
    if usage is None or not usage.ru_maxrss:
        return None
    # Linux reports kilobytes and macOS bytes.
    if sys.platform == "darwin":
        return usage.ru_maxrss / 1e6
    return usage.ru_maxrss / 1e3


def _read_and_remove(log_path: str) -> str:
    try:
        with open(log_path, encoding="utf-8", errors="replace") as log:
            return log.read()
    except FileNotFoundError:
        return ""
    finally:
        try:
            os.remove(log_path)
        except FileNotFoundError:
            pass
