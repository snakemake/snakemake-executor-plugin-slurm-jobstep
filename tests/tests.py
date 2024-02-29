from typing import Optional
import snakemake.common.tests
from snakemake_interface_executor_plugins.settings import ExecutorSettingsBase


class TestWorkflowsBase(snakemake.common.tests.TestWorkflowsLocalStorageBase):
    __test__ = True

    def get_executor(self) -> str:
        return "slurm-jobstep"

    def get_executor_settings(self) -> Optional[ExecutorSettingsBase]:
        # instatiate ExecutorSettings of this plugin as appropriate
        return None


@skip_on_windows
def test_issue_41():
    run(dpath("test_github_issue41"))
