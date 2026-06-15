from snakemake_executor_plugin_slurm_jobstep.stagein import expand_node_local_prefix


def test_expand_node_local_prefix_replaces_env_markers(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    value = "/localscratch/__ENV_SLURM_JOB_ID__/run"
    assert expand_node_local_prefix(value) == "/localscratch/12345/run"


def test_expand_node_local_prefix_leaves_plain_paths_unchanged(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    value = "/localscratch/run"
    assert expand_node_local_prefix(value) == "/localscratch/run"
