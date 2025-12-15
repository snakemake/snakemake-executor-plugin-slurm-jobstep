__author__ = "David Lähnemann, Johannes Köster, Christian Meesters"
__copyright__ = "Copyright 2023, David Lähnemann, Johannes Köster, Christian Meesters"
__email__ = "johannes.koester@uni-due.de"
__license__ = "MIT"

import os
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from snakemake_interface_executor_plugins.executors.base import SubmittedJobInfo
from snakemake_interface_executor_plugins.executors.real import RealExecutor
from snakemake_interface_executor_plugins.jobs import (
    JobExecutorInterface,
)
from snakemake_interface_executor_plugins.settings import (
    CommonSettings,
    ExecMode,
    ExecutorSettingsBase,
)
from snakemake_interface_common.exceptions import WorkflowError


# Required:
# Specify common settings shared by various executors.
common_settings = CommonSettings(
    # define whether your executor plugin executes locally
    # or remotely. In virtually all cases, it will be remote execution
    # (cluster, cloud, etc.). Only Snakemake's standard execution
    # plugins (snakemake-executor-plugin-dryrun, snakemake-executor-plugin-local)
    # are expected to specify False here.
    non_local_exec=True,
    # Define whether your executor plugin implies that there is no shared
    # filesystem (True) or not (False).
    # This is e.g. the case for cloud execution.
    implies_no_shared_fs=False,
    job_deploy_sources=False,
    pass_default_storage_provider_args=True,
    pass_default_resources_args=True,
    pass_envvar_declarations_to_cmd=False,
    auto_deploy_default_storage_provider=False,
    spawned_jobs_assume_shared_fs=True,
)


@dataclass
class ExecutorSettings(ExecutorSettingsBase):
    """Settings for the SLURM jobstep executor plugin."""

    pass_command_as_script: bool = field(
        default=False,
        metadata={
            "help": (
                "Pass to srun the command to be executed as a shell script "
                "(fed through stdin) instead of wrapping it in the command line "
                "call. Useful when a limit exists on SLURM command line length (ie. "
                "max_submit_line_size)."
            ),
            "env_var": False,
            "required": False,
        },
    )


# Required:
# Implementation of your executor
class Executor(RealExecutor):
    def __post_init__(self):
        # These environment variables are set by SLURM.
        # only needed for commented out jobstep handling below
        self.jobid = os.getenv("SLURM_JOB_ID")
        # we consider this job to be a GPU job, if a GPU has been reserved
        self.gpu_job = os.getenv("SLURM_GPUS")

    def run_job(self, job: JobExecutorInterface):
        # Implement here how to run a job.
        # You can access the job's resources, etc.
        # via the job object.
        # After submitting the job, you have to call
        # self.report_job_submission(job_info).
        # with job_info being of type
        # snakemake_interface_executor_plugins.executors.base.SubmittedJobInfo.

        jobsteps = dict()
        srun_script = None
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
        if "mpi" in job.resources.keys():
            # MPI job:
            # No need to prepend `srun`, as this will happen inside of the job's shell
            # command or script (!).
            # The following call invokes snakemake, which in turn takes care of all
            # auxiliary work around the actual command
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

            # as of v22.11.0, the --cpu-per-task flag is needed to ensure that
            # the job can utilize the c-group's resources.
            # We set the limitation accordingly, assuming the submit executor
            # has set the resources correctly.

            call = "srun -n1 --cpu-bind=q "
            call += f" {get_cpu_setting(job, self.gpu_job)} "
            if self.workflow.executor_settings.pass_command_as_script:
                # format the job to execute with all the snakemake parameters
                # into a script
                srun_script = self.format_job_exec(job)
                # the process will read the srun script from stdin
                call += " sh -s"
            else:
                call += f" {self.format_job_exec(job)}"

        self.logger.debug(f"This job is a group job: {job.is_group()}")
        self.logger.debug(f"The call for this job is: {call}")
        self.logger.debug(f"Job is running on host: {socket.gethostname()}")
        if srun_script is not None:
            self.logger.debug(f"The script for this job is: \n{srun_script}")
        # this dict is to support the to be implemented feature of oversubscription in
        # "ordinary" group jobs.
        jobsteps[job] = subprocess.Popen(
            call, shell=True, text=True, stdin=subprocess.PIPE
        )
        if srun_script is not None:
            try:
                # pass the srun bash script via stdin
                jobsteps[job].stdin.write(srun_script)
                jobsteps[job].stdin.close()
            except BrokenPipeError:
                # subprocess terminated before reading stdin
                self.logger.error(
                    f"Failed to write script to stdin for job {job}. "
                    "Subprocess may have terminated prematurely."
                )
                self.report_job_error(SubmittedJobInfo(job))
                raise WorkflowError(
                    f"Job {job} failed: subprocess terminated before reading script"
                )

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

    @property
    def cores(self):
        return "all"

    def get_exec_mode(self) -> ExecMode:
        return ExecMode.REMOTE


def get_cpu_setting(job: JobExecutorInterface, gpu: bool) -> str:
    # per default, we assume that Snakemake's threads are the same as the
    # cpus per task or per gpu. If the user has set the cpus_per_task or
    # cpus_per_gpu explicitly, we use these values.
    cpus_per_task = cpus_per_gpu = job.threads
    # cpus_per_task and cpus_per_gpu are mutually exclusive
    if job.resources.get("cpus_per_task"):
        cpus_per_task = job.resources.cpus_per_task
        if not isinstance(cpus_per_task, int):
            raise WorkflowError(
                f"cpus_per_task must be an integer, but is {cpus_per_task}"
            )
        # If explicetily set to < 0, return an empty string
        # some clusters do not allow CPU settings (e.g. in GPU partitions).
        if cpus_per_task < 0:
            return ""
        # ensure that at least 1 cpu is requested
        # because 0 is not allowed by slurm
        cpus_per_task = max(1, job.resources.cpus_per_task)
        return f"--cpus-per-task={cpus_per_task}"
    elif gpu and job.resources.get("cpus_per_gpu"):
        cpus_per_gpu = job.resources.cpus_per_gpu
        if not isinstance(cpus_per_gpu, int):
            raise WorkflowError(
                f"cpus_per_gpu must be an integer, but is {cpus_per_gpu}"
            )
        # If explicetily set to < 0, return an empty string
        # some clusters do not allow CPU settings (e.g. in GPU partitions).
        # Currently, 0 is not allowed by SLURM.
        if cpus_per_gpu <= 0:
            return ""
        return f"--cpus-per-gpu={cpus_per_gpu}"
    else:
        return f"--cpus-per-task={cpus_per_task}"
