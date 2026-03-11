from typing import Optional
import os
import pytest
import snakemake.common.tests
from snakemake_interface_executor_plugins.settings import ExecutorSettingsBase
from snakemake_interface_common.exceptions import WorkflowError

from snakemake_executor_plugin_slurm_jobstep import ExecutorSettings, parse_array_execs


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


def test_parse_array_execs_invalid_value_raises():
    with pytest.raises(WorkflowError, match="hex-encoded"):
        parse_array_execs('{"2": "not-hex"}')
