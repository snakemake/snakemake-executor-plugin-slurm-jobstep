from typing import Optional
import os
import base64
import signal
import sys
from pathlib import Path
import zlib
import pytest
import snakemake.common.tests
from snakemake_interface_executor_plugins.settings import ExecutorSettingsBase
from snakemake_interface_common.exceptions import WorkflowError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from snakemake_executor_plugin_slurm_jobstep import (
    ExecutorSettings,
    _decompress_array_task_call,
    _forward_signal_to_non_snakemake_descendants,
    _is_python_cmdline,
    _is_snakemake_cmdline,
    _is_first_array_task,
    _parse_trapped_signal,
    parse_array_execs,
    strip_array_execs_option,
)


# Inserted for local test for utility functions, which do not require SLURM
@pytest.mark.skipif(
    os.getenv("SLURM_JOB_ID") is None,
    reason="Workflow integration tests require running inside a SLURM allocation.",
)
class TestWorkflowsBase(snakemake.common.tests.TestWorkflowsLocalStorageBase):
    __test__ = True

    def get_executor(self) -> str:
        return "slurm-jobstep"

    def get_executor_settings(self) -> Optional[ExecutorSettingsBase]:
        # instatiate ExecutorSettings of this plugin as appropriate
        return ExecutorSettings()


# def test_issue_41():
#    run(dpath("test_github_issue41"))


def test_parse_array_execs_json():
    parsed = parse_array_execs('{"2": "a1b2", "6": "deadbeef"}')
    assert parsed == {"2": "a1b2", "6": "deadbeef"}


def test_parse_array_execs_python_literal():
    parsed = parse_array_execs("{2: 'a1b2', 6: 'deadbeef'}")
    assert parsed == {"2": "a1b2", "6": "deadbeef"}


def test_parse_array_execs_compact_mapping():
    parsed = parse_array_execs("{2: a1b2, 6: deadbeef}")
    assert parsed == {"2": "a1b2", "6": "deadbeef"}


def test_parse_array_execs_base64_json_unquoted():
    raw = '{"2": "a1b2", "6": "deadbeef"}'
    encoded = base64.b64encode(raw.encode("utf-8")).decode("ascii")
    parsed = parse_array_execs(encoded)
    assert parsed == {"2": "a1b2", "6": "deadbeef"}


def test_parse_array_execs_base64_json_single_quoted():
    raw = '{"2": "a1b2", "6": "deadbeef"}'
    encoded = base64.b64encode(raw.encode("utf-8")).decode("ascii")
    parsed = parse_array_execs(f"'{encoded}'")
    assert parsed == {"2": "a1b2", "6": "deadbeef"}


def test_parse_array_execs_invalid_value_raises():
    with pytest.raises(WorkflowError, match="hex-encoded"):
        parse_array_execs('{"2": "not-hex"}')


def test_strip_array_execs_option_equals_form():
    cmd = (
        "python -m snakemake --executor slurm-jobstep "
        "--slurm-jobstep-array-execs='{" + '"2": "a1b2"' + "}' --jobs 1"
    )
    stripped = strip_array_execs_option(cmd)
    assert "--slurm-jobstep-array-execs" not in stripped
    assert "--executor slurm-jobstep" in stripped
    assert "--jobs 1" not in stripped


def test_strip_array_execs_option_base64_quoted_form():
    payload = "eyIyIjogImExYjIiLCAiNiI6ICJkZWFkYmVlZiJ9"
    cmd = (
        "python -m snakemake --executor slurm-jobstep "
        f"--slurm-jobstep-array-execs='{payload}' --jobs 1"
    )
    stripped = strip_array_execs_option(cmd)
    assert "--slurm-jobstep-array-execs" not in stripped
    assert "--executor slurm-jobstep" in stripped
    assert "--jobs 1" not in stripped


def test_strip_array_execs_option_base64_unquoted_form():
    payload = "eyIyIjogImExYjIiLCAiNiI6ICJkZWFkYmVlZiJ9"
    cmd = (
        "python -m snakemake --executor slurm-jobstep "
        f"--slurm-jobstep-array-execs={payload} --jobs 1"
    )
    stripped = strip_array_execs_option(cmd)
    assert "--slurm-jobstep-array-execs" not in stripped
    assert "--executor slurm-jobstep" in stripped
    assert "--jobs 1" not in stripped


def test_strip_array_execs_option_separate_form():
    cmd = (
        "python -m snakemake --executor slurm-jobstep "
        "--slurm-jobstep-array-execs '{2: a1b2, 3: deadbeef}' --jobs 1"
    )
    stripped = strip_array_execs_option(cmd)
    assert "--slurm-jobstep-array-execs" not in stripped
    assert "--executor slurm-jobstep" in stripped
    assert "--jobs 1" not in stripped


def test_is_first_array_task_uses_task_min(monkeypatch):
    monkeypatch.setenv("SLURM_ARRAY_TASK_MIN", "5")
    assert _is_first_array_task(5)
    assert not _is_first_array_task(6)


def test_is_first_array_task_missing_task_min(monkeypatch):
    monkeypatch.delenv("SLURM_ARRAY_TASK_MIN", raising=False)
    assert not _is_first_array_task(1)


def test_is_first_array_task_invalid_task_min_raises(monkeypatch):
    monkeypatch.setenv("SLURM_ARRAY_TASK_MIN", "x")
    with pytest.raises(WorkflowError, match="SLURM_ARRAY_TASK_MIN"):
        _is_first_array_task(1)


def test_decompress_array_task_call_missing_index_raises():
    compressed = zlib.compress(b"echo hi").hex()
    with pytest.raises(WorkflowError, match="Missing compressed array command"):
        _decompress_array_task_call('{"2": "' + compressed + '"}', 3)


def test_decompress_array_task_call_valid_payload():
    expected = "echo hello"
    compressed = zlib.compress(expected.encode("utf-8")).hex()
    resolved = _decompress_array_task_call('{"2": "' + compressed + '"}', 2)
    assert resolved == expected


def test_parse_trapped_signal_with_time_suffix():
    assert _parse_trapped_signal("12@60") == 12
    assert _parse_trapped_signal("15@120") == 15


def test_parse_trapped_signal_without_time_suffix():
    assert _parse_trapped_signal("12") == 12


def test_parse_trapped_signal_with_slurm_batch_prefix():
    assert _parse_trapped_signal("B:23@60") == 23


def test_parse_trapped_signal_empty_is_none():
    assert _parse_trapped_signal(None) is None
    assert _parse_trapped_signal("   ") is None


def test_parse_trapped_signal_invalid_raises():
    with pytest.raises(WorkflowError, match="Invalid signal setting"):
        _parse_trapped_signal("SIGUSR2@60")


def test_is_snakemake_cmdline_true_for_snakemake_invocations():
    assert _is_snakemake_cmdline("python -m snakemake --cores 1")
    assert _is_snakemake_cmdline("/usr/bin/snakemake --executor slurm")


def test_is_snakemake_cmdline_false_for_non_snakemake_cmdlines():
    assert not _is_snakemake_cmdline("/usr/bin/bash -lc bwa mem ref.fa reads.fq")


def test_is_python_cmdline_true_for_python_invocations():
    assert _is_python_cmdline("python -m module")
    assert _is_python_cmdline("/usr/bin/python3.13 -m snakemake")


def test_is_python_cmdline_false_for_non_python_invocations():
    assert not _is_python_cmdline("/usr/bin/bash -lc bwa mem ref.fa reads.fq")
    assert not _is_python_cmdline("/opt/bin/mypython-wrapper run")


def test_forward_signal_forwards_all_descendants(monkeypatch):
    """Verify signals propagate through all descendants, including nested snakemake."""
    class _DummyLogger:
        def debug(self, *_args, **_kwargs):
            pass

        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

    class _DummyProc:
        pid = 100

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(
        "snakemake_executor_plugin_slurm_jobstep._get_descendant_pids",
        lambda _pid: {101, 102, 103},
    )

    cmdlines = {
        101: "/usr/bin/python3 -m snakemake --cores 1",
        102: "/usr/bin/python3 worker.py",
        103: "/usr/bin/bash -lc sleep 60",
    }
    monkeypatch.setattr(
        "snakemake_executor_plugin_slurm_jobstep._read_cmdline",
        lambda pid: cmdlines[pid],
    )

    killed = []

    def _fake_kill(pid, signum):
        killed.append((pid, signum))

    monkeypatch.setattr(os, "kill", _fake_kill)

    forwarded = _forward_signal_to_non_snakemake_descendants(
        _DummyProc(), signal.SIGURG, _DummyLogger()
    )

    # All descendants receive the signal, including nested snakemake/python
    assert forwarded == 3
    assert killed == [(101, signal.SIGURG), (102, signal.SIGURG), (103, signal.SIGURG)]


def test_forward_signal_uses_process_group_fallback(monkeypatch):
    class _DummyLogger:
        def debug(self, *_args, **_kwargs):
            pass

        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

    class _DummyProc:
        pid = 200

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(
        "snakemake_executor_plugin_slurm_jobstep._get_descendant_pids",
        lambda _pid: set(),
    )
    monkeypatch.setattr(
        "snakemake_executor_plugin_slurm_jobstep._get_same_process_group_pids",
        lambda _pid: {200, 201, 202},
    )

    cmdlines = {
        201: "/usr/bin/python3 -m snakemake --cores 1",
        202: "/usr/bin/bash -lc sleep 60",
    }
    monkeypatch.setattr(
        "snakemake_executor_plugin_slurm_jobstep._read_cmdline",
        lambda pid: cmdlines.get(pid, ""),
    )

    killed = []

    def _fake_kill(pid, signum):
        killed.append((pid, signum))

    monkeypatch.setattr(os, "kill", _fake_kill)

    forwarded = _forward_signal_to_non_snakemake_descendants(
        _DummyProc(), signal.SIGURG, _DummyLogger()
    )

    # Now forwards to all, including the nested snakemake
    assert forwarded == 2
    assert killed == [(201, signal.SIGURG), (202, signal.SIGURG)]


def test_forward_signal_includes_user_process_candidates(monkeypatch):
    class _DummyLogger:
        def debug(self, *_args, **_kwargs):
            pass

        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

    class _DummyProc:
        pid = 400

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(
        "snakemake_executor_plugin_slurm_jobstep._get_descendant_pids",
        lambda _pid: {403},
    )
    monkeypatch.setattr(
        "snakemake_executor_plugin_slurm_jobstep._get_same_process_group_pids",
        lambda _pid: set(),
    )
    monkeypatch.setattr(
        "snakemake_executor_plugin_slurm_jobstep._get_user_process_pids",
        lambda _uid: {401, 402},
    )

    cmdlines = {
        403: "/usr/bin/bash -lc true",
        401: "/usr/bin/python3 -m snakemake --cores 1",
        402: "/usr/bin/bash -lc sleep 60",
    }
    monkeypatch.setattr(
        "snakemake_executor_plugin_slurm_jobstep._read_cmdline",
        lambda pid: cmdlines.get(pid, ""),
    )

    killed = []

    def _fake_kill(pid, signum):
        killed.append((pid, signum))

    monkeypatch.setattr(os, "kill", _fake_kill)

    forwarded = _forward_signal_to_non_snakemake_descendants(
        _DummyProc(), signal.SIGURG, _DummyLogger()
    )

    # Now forwards to all candidates, including python and bash processes
    assert forwarded == 3
    assert killed == [(401, signal.SIGURG), (402, signal.SIGURG), (403, signal.SIGURG)]
