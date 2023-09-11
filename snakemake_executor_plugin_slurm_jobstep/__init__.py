__author__ = "David Lähnemann, Johannes Köster, Christian Meesters"
__copyright__ = "Copyright 2023, David Lähnemann, Johannes Köster, Christian Meesters"
__email__ = "johannes.koester@uni-due.de"
__license__ = "MIT"

import os
import subprocess
import sys
from snakemake_interface_executor_plugins.executors.base import SubmittedJobInfo
from snakemake_interface_executor_plugins.executors.real import RealExecutor
from snakemake_interface_executor_plugins import CommonSettings
from snakemake_interface_executor_plugins.workflow import WorkflowExecutorInterface
from snakemake_interface_executor_plugins.logging import LoggerExecutorInterface
from snakemake_interface_executor_plugins.jobs import (
    ExecutorJobInterface,
)
from snakemake_interface_executor_plugins.settings import ExecMode


# Required:
# Specify common settings shared by various executors.
common_settings = CommonSettings(
    # define whether your executor plugin executes locally
    # or remotely. In virtually all cases, it will be remote execution
    # (cluster, cloud, etc.). Only Snakemake's standard execution
    # plugins (snakemake-executor-plugin-dryrun, snakemake-executor-plugin-local)
    # are expected to specify False here.
    non_local_exec=False,
    # Define whether your executor plugin implies that there is no shared
    # filesystem (True) or not (False).
    # This is e.g. the case for cloud execution.
    implies_no_shared_fs=False,
)


# Required:
# Implementation of your executor
class Executor(RealExecutor):
    def __init__(
        self,
        workflow: WorkflowExecutorInterface,
        logger: LoggerExecutorInterface,
    ):
        super().__init__(
            workflow,
            logger,
            # configure behavior of RemoteExecutor below
            # whether arguments for setting the remote provider shall  be passed to jobs
            pass_default_remote_provider_args=True,
            # whether arguments for setting default resources shall be passed to jobs
            pass_default_resources_args=True,
            # whether environment variables shall be passed to jobs
            pass_envvar_declarations_to_cmd=False,
        )
        # These environment variables are set by SLURM.
        # only needed for commented out jobstep handling below
        self.jobid = os.getenv("SLURM_JOB_ID")

    def run_job(self, job: ExecutorJobInterface):
        # Implement here how to run a job.
        # You can access the job's resources, etc.
        # via the job object.
        # After submitting the job, you have to call
        # self.report_job_submission(job_info).
        # with job_info being of type
        # snakemake_interface_executor_plugins.executors.base.SubmittedJobInfo.

        jobsteps = dict()
        # TODO revisit special handling for group job levels via srun at a later stage
        # if job.is_group():

        #     def get_call(level_job, aux=""):
        #         # we need this calculation, because of srun's greediness and
        #         # SLURM's limits: it is not able to limit the memory if we divide the
        #         # job per CPU by itself.

        #         level_mem = level_job.resources.get("mem_mb")
        #         if isinstance(level_mem, TBDString):
        #             level_mem = 100

        #         mem_per_cpu = max(level_mem // level_job.threads, 100)
        #         exec_job = self.format_job_exec(level_job)

        #         # Note: The '--exlusive' flag is a prevention for triggered job steps
        #         #       within an allocation to oversubscribe within a given c-group.
        #         #       As we are dealing only with smp software
        #         #       the '--ntasks' is explicitly set to 1 by '-n1' per group job
        #         #       (step).
        #         return (
        #             f"srun -J {job.groupid} --jobid {self.jobid}"
        #             f" --mem-per-cpu {mem_per_cpu} -c {level_job.threads}"
        #             f" --exclusive -n 1 {aux} {exec_job}"
        #         )

        #     for level in list(job.toposorted):
        #         # we need to ensure order - any:
        #         level_list = list(level)
        #         for level_job in level_list[:-1]:
        #             jobsteps[level_job] = subprocess.Popen(
        #                 get_call(level_job), shell=True
        #             )
        #         # now: the last one
        #         # this way, we ensure that level jobs depending on the current level
        #         # get started
        #         jobsteps[level_list[-1]] = subprocess.Popen(
        #             get_call(level_list[-1], aux="--dependency=singleton"), shell=True
        #         )

        if "mpi" in job.resources.keys():
            # MPI job:
            # No need to prepend `srun`, as this will happen inside of the job's shell
            # command or script (!).
            # The following call invokes snakemake, which in turn takes care of all
            # auxilliary work around the actual command
            # like remote file support, benchmark setup, error handling, etc.
            # AND there can be stuff around the srun call within the job, like any
            # commands which should be executed before.
            call = self.format_job_exec(job)
        else:
            # SMP job, execute snakemake with srun, to ensure proper placing of threaded
            # executables within the c-group
            # The -n1 is important to avoid that srun executes the given command
            # multiple times, depending on the relation between
            # cpus per task and the number of CPU cores.
            call = f"srun -n1 --cpu-bind=q {self.format_job_exec(job)}"

        # this dict is to support the to be implemented feature of oversubscription in
        # "ordinary" group jobs.
        jobsteps[job] = subprocess.Popen(call, shell=True)

        job_info = SubmittedJobInfo(job)
        self.report_job_submission(job_info)

        # wait until all steps are finished
        if any(proc.wait() != 0 for proc in jobsteps.values()):
            self.report_job_error(job_info)
        else:
            self.report_job_success(job_info)

    def cancel(self):
        pass

    def shutdown(self):
        pass

    def get_python_executable(self):
        return sys.executable

    def get_envvar_declarations(self):
        return ""

    @property
    def cores(self):
        return "all"

    def get_exec_mode(self) -> ExecMode:
        return ExecMode.REMOTE
