import os
from typing import Optional
import snakemake.common.tests
from snakemake_interface_executor_plugins import ExecutorSettingsBase


# fake slurm job environment
os.environ["SLURM_JOB_ID"] = "100"
os.environ["SLURM_MEM_PER_NODE"] = "100"
os.environ["SLURM_CPUS_ON_NODE"] = "1"


class TestWorkflowsBase(snakemake.common.tests.TestWorkflowsBase):
    __test__ = True

    def get_executor(self) -> str:
        return "slurm-jobstep"

    def get_executor_settings(self) -> Optional[ExecutorSettingsBase]:
        # instatiate ExecutorSettings of this plugin as appropriate
        return None

    def get_default_remote_provider(self) -> Optional[str]:
        # Return name of default remote provider if required for testing,
        # otherwise None.
        return None

    def get_default_remote_prefix(self) -> Optional[str]:
        # Return default remote prefix if required for testing,
        # otherwise None.
        return None
