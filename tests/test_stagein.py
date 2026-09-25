import subprocess
from types import SimpleNamespace
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from snakemake_interface_common.exceptions import WorkflowError

import snakemake_executor_plugin_slurm_jobstep.stagein as stagein
from snakemake_executor_plugin_slurm_jobstep.stagein import (
    get_nodelist,
)


def test_get_nodelist_raises_when_slurm_nodelist_missing(monkeypatch):
    monkeypatch.delenv("SLURM_NODELIST", raising=False)
    with pytest.raises(WorkflowError, match="SLURM_NODELIST"):
        get_nodelist()


def test_get_nodelist_raises_when_slurm_nodelist_empty(monkeypatch):
    monkeypatch.setenv("SLURM_NODELIST", "")
    with pytest.raises(WorkflowError, match="SLURM_NODELIST"):
        get_nodelist()


def test_get_nodelist_raises_when_scontrol_returns_no_hosts(monkeypatch):
    monkeypatch.setenv("SLURM_NODELIST", "compute[01-02]")
    monkeypatch.setattr(
        stagein.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="\n"),
    )
    with pytest.raises(WorkflowError, match="no nodes were returned"):
        get_nodelist()


def test_get_nodelist_raises_when_scontrol_fails(monkeypatch):
    monkeypatch.setenv("SLURM_NODELIST", "compute[01-02]")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=args[0],
        )

    monkeypatch.setattr(stagein.subprocess, "run", fail)
    with pytest.raises(WorkflowError, match="Failed to expand SLURM nodelist"):
        get_nodelist()


def test_get_nodename_raises_when_empty(monkeypatch):
    monkeypatch.setenv("SLURMD_NODENAME", "")
    with pytest.raises(WorkflowError, match="SLURMD_NODENAME"):
        stagein.get_nodename()
