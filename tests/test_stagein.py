import pytest
from snakemake_interface_common.exceptions import WorkflowError

from snakemake_executor_plugin_slurm_jobstep.stagein import (
    expand_node_local_prefix,
    get_nodelist,
)


def test_expand_node_local_prefix_replaces_env_markers(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    value = "/localscratch/__ENV_SLURM_JOB_ID__/run"
    assert expand_node_local_prefix(value) == "/localscratch/12345/run"


def test_expand_node_local_prefix_leaves_plain_paths_unchanged(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    value = "/localscratch/run"
    assert expand_node_local_prefix(value) == "/localscratch/run"


def test_expand_node_local_prefix_raises_for_missing_env_marker(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(WorkflowError, match="SLURM_JOB_ID is not set"):
        expand_node_local_prefix("/localscratch/__ENV_SLURM_JOB_ID__/run")


def test_get_nodelist_raises_when_slurm_nodelist_missing(monkeypatch):
    monkeypatch.delenv("SLURM_NODELIST", raising=False)
    with pytest.raises(WorkflowError, match="SLURM_NODELIST"):
        get_nodelist()
