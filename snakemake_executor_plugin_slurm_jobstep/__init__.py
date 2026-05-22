__author__ = "David Lähnemann, Johannes Köster, Christian Meesters"
__copyright__ = "Copyright 2023, David Lähnemann, Johannes Köster, Christian Meesters"
__email__ = "johannes.koester@uni-due.de"
__license__ = "MIT"

import base64
import binascii
import os
import signal
import socket
import subprocess
import sys
import json
import ast
import re
import zlib
import pwd
from dataclasses import dataclass, field
from typing import Optional
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

    pass_command_as_script: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Pass to srun the command to be executed as a shell script "
                "(fed through stdin) instead of wrapping it in the command line "
                "call. Useful when a limit exists on SLURM command line length (ie. "
                "max_submit_line_size). (internal use only)"
            ),
            "env_var": False,
            "required": False,
        },
    )
    array_execs: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "When a job array is used, this flag, will receive all job excec "
                "strings as a json dict. (internal use only)"
            ),
            "env_var": False,
            "required": False,
        },
    )
    signal: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Internal use only: The signal to trap and forward to job steps."
                "If not set, no signal forwarding will be performed."
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
        # check if SLURM_ARRAY_TASK_ID is set, to determine whether this
        # is a job array task
        self.job_array_task = os.getenv("SLURM_ARRAY_TASK_ID") is not None
        self.signal_setting = self.workflow.executor_settings.signal
        self.logger.debug(
            f"Executor initialized with signal setting: {self.signal_setting}"
        )

    def run_job(self, job: JobExecutorInterface):
        # Implement here how to run a job.
        # You can access the job's resources, etc.
        # via the job object.
        # After submitting the job, you have to call
        # self.report_job_submission(job_info).
        # with job_info being of type
        # snakemake_interface_executor_plugins.executors.base.SubmittedJobInfo.

        jobsteps = dict()
        call = None
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
        srun_signal_setting = _get_srun_signal_setting(self.signal_setting)

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
            if srun_signal_setting and re.match(r"^\s*srun\b", call):
                call = re.sub(
                    r"^\s*srun\b",
                    f"srun --signal={srun_signal_setting}",
                    call,
                    count=1,
                )
        # this is an array job
        elif self.job_array_task and self.workflow.executor_settings.array_execs:
            array_index = int(os.getenv("SLURM_ARRAY_TASK_ID"))
            call = "srun -n1 --cpu-bind=q "
            if srun_signal_setting:
                call += f"--signal={srun_signal_setting} "
            call += f" {get_cpu_setting(job, self.gpu_job)} "
            if _is_first_array_task(array_index):
                raw_call = self.format_job_exec(job)
                call += strip_array_execs_option(raw_call)
                self.logger.debug(
                    f"Using raw call for first array task index {array_index}: {call}"
                )
            else:
                call += _decompress_array_task_call(
                    self.workflow.executor_settings.array_execs,
                    array_index,
                )
                self.logger.debug(
                    f"Decompressed call for array index {array_index}: {call}"
                )
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
            if srun_signal_setting:
                call += f"--signal={srun_signal_setting} "
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

        previous_handlers: dict[int, object] = {}
        trapped_signal = _parse_trapped_signal(self.signal_setting)
        if trapped_signal is not None:
            self.logger.info(f"Signal forwarding enabled for signal {trapped_signal}.")

        signal_forwarded = False
        pending_signal: Optional[int] = None

        def _forward_handler(received_signal, _frame):
            nonlocal signal_forwarded, pending_signal
            if signal_forwarded:
                self.logger.debug(
                    f"Signal {received_signal} received again; signal has already been forwarded once."
                )
                return

            self.logger.info(
                f"Received signal {received_signal}, forwarding to non-Snakemake descendant processes."
            )
            proc = jobsteps.get(job)
            if proc is None:
                pending_signal = received_signal
                self.logger.info(
                    f"Received signal {received_signal} before job process became available; postponing forwarding."
                )
                return
            if proc.poll() is not None:
                self.logger.warning(
                    f"Job process PID {proc.pid} already terminated; cannot forward signal {received_signal}."
                )
                return

            signal_forwarded = True
            self.logger.debug(f"Job process PID: {proc.pid}, poll status: {proc.poll()}")
            forwarded_count = _forward_signal_to_non_snakemake_descendants(
                proc, received_signal, self.logger
            )
            self.logger.info(
                f"Forwarded signal {received_signal} to {forwarded_count} non-Snakemake, non-Python processes."
            )

        if trapped_signal is not None:
            try:
                previous_handlers[trapped_signal] = signal.getsignal(trapped_signal)
                signal.signal(trapped_signal, _forward_handler)
                self.logger.info(
                    f"Registered forwarding handler for signal {trapped_signal}."
                )
            except (OSError, RuntimeError, ValueError) as err:
                raise WorkflowError(
                    f"Failed to register signal handler for signal {trapped_signal}."
                ) from err

        # this dict is to support the to be implemented feature of oversubscription in
        # "ordinary" group jobs.
        jobsteps[job] = subprocess.Popen(
            call,
            shell=True,
            text=True,
            stdin=subprocess.PIPE,
            # Keep all descendants in one process group so signals can be forwarded.
            start_new_session=True,
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

        if pending_signal is not None and not signal_forwarded:
            self.logger.info(
                f"Processing postponed forwarding for signal {pending_signal}."
            )
            _forward_handler(pending_signal, None)

        job_info = SubmittedJobInfo(job)
        self.report_job_submission(job_info)

        try:
            # wait until all steps are finished
            if any(proc.wait() != 0 for proc in jobsteps.values()):
                self.report_job_error(job_info)
            else:
                self.report_job_success(job_info)
        finally:
            if trapped_signal is not None:
                for signum, previous_handler in previous_handlers.items():
                    try:
                        signal.signal(signum, previous_handler)
                    except (OSError, RuntimeError, ValueError):
                        # Best effort restore; process is about to finish anyway.
                        pass

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


def parse_array_execs(raw_array_execs) -> dict:
    """Parse array exec mapping from executor settings.

    Accepts strict JSON and Python literal dict strings for compatibility with
    shell/CLI forwarding that may rewrite quotes.
    """
    if isinstance(raw_array_execs, dict):
        parsed = raw_array_execs

    elif not isinstance(raw_array_execs, str):
        raise WorkflowError(
            "Invalid value for executor setting `array_execs`: expected str or dict."
        )
    else:
        candidate = raw_array_execs.strip()
        if (
            len(candidate) >= 2
            and candidate[0] == candidate[-1]
            and candidate[0] in ("'", '"')
        ):
            candidate = candidate[1:-1]
        try:
            parsed = json.loads(base64.b64decode(candidate))
        except (json.JSONDecodeError, binascii.Error, ValueError, TypeError):
            try:
                parsed = ast.literal_eval(raw_array_execs)
            except (SyntaxError, ValueError):
                parsed = _parse_compact_array_execs(raw_array_execs)
                if parsed is None:
                    raise WorkflowError(
                        "Failed to parse executor setting `array_execs`. "
                        "Expected JSON, Python dict literal, or compact mapping."
                    ) from None

    if not isinstance(parsed, dict):
        raise WorkflowError(
            "Invalid value for executor setting `array_execs`: expected mapping."
        )

    normalized = {}
    for key, value in parsed.items():
        key_str = str(key).strip()
        if not key_str:
            raise WorkflowError(
                "Invalid value for executor setting `array_execs`: empty task id key."
            )

        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if not value or not re.fullmatch(r"[0-9a-fA-F]+", value):
            raise WorkflowError(
                "Invalid value for executor setting `array_execs`: values must be "
                "hex-encoded strings."
            )
        normalized[key_str] = value

    return normalized


def strip_array_execs_option(command: str) -> str:
    """Strip --slurm-jobstep-array-execs and all trailing arguments.

    Truncates the command at the first occurrence of --slurm-jobstep-array-execs
    (in either --flag=value or --flag value form).
    """
    match = re.search(r"\s*--slurm-jobstep-array-execs(?:=|\s)", command)
    if match:
        return command[: match.start()].rstrip()
    return command


def _is_first_array_task(array_index: int) -> bool:
    """Return whether current task is the first job array task.

    Prefers SLURM_ARRAY_TASK_MIN to avoid parsing potentially large array_exec
    mappings in first tasks.
    """
    task_min = os.getenv("SLURM_ARRAY_TASK_MIN")
    if task_min is None:
        return False

    try:
        return array_index == int(task_min)
    except ValueError as err:
        raise WorkflowError(
            f"Invalid SLURM_ARRAY_TASK_MIN value: {task_min!r}."
        ) from err


def _decompress_array_task_call(raw_array_execs: str, array_index: int) -> str:
    """Return decompressed command for one array task index."""
    array_execs = parse_array_execs(raw_array_execs)
    compressed_hex = array_execs.get(str(array_index))
    if compressed_hex is None:
        raise WorkflowError(
            "Missing compressed array command for task index "
            f"{array_index} in executor setting `array_execs`."
        )

    compressed_bytes = bytes.fromhex(compressed_hex)
    return zlib.decompress(compressed_bytes).decode("utf-8")


def _parse_trapped_signal(signal_setting: Optional[str]) -> Optional[int]:
    """Extract signal number from '<signal>@<seconds>' or 'B:<signal>@<seconds>'."""
    if signal_setting is None:
        return None

    signal_text = signal_setting.strip()
    if not signal_text:
        return None

    signal_spec, _, _ = signal_text.partition("@")
    signal_spec = signal_spec.strip()
    if signal_spec.upper().startswith("B:"):
        signal_spec = signal_spec[2:].strip()

    try:
        return int(signal_spec)
    except ValueError as err:
        raise WorkflowError(
            "Invalid signal setting: expected '<signal number>@<seconds>', "
            "'<signal number>', or 'B:<signal number>@<seconds>'."
        ) from err


def _get_srun_signal_setting(signal_setting: Optional[str]) -> Optional[str]:
    """Return normalized signal setting string for use with srun --signal."""
    if signal_setting is None:
        return None

    setting = signal_setting.strip()
    if not setting:
        return None

    if setting.upper().startswith("B:"):
        setting = setting[2:].strip()

    return setting or None


def _forward_signal_to_non_snakemake_descendants(
    process: Optional[subprocess.Popen], signum: int, logger
) -> int:
    """Forward a signal to non-Snakemake, non-Python related processes."""
    if process is None or process.poll() is not None:
        logger.debug(f"Process is None or already terminated.")
        return 0

    root_pid = process.pid
    logger.debug(f"Scanning descendants and process-group peers of PID {root_pid}.")
    descendants = _get_descendant_pids(root_pid)
    group_peers = _get_same_process_group_pids(root_pid)
    user_candidates = _get_user_process_pids(os.getuid())
    candidates = descendants.union(group_peers).union(user_candidates)
    logger.debug(
        f"Found {len(descendants)} descendant PIDs: {sorted(descendants)}; "
        f"{len(group_peers)} same-group PIDs: {sorted(group_peers)}; "
        f"{len(user_candidates)} user-candidate PIDs: {sorted(user_candidates)}; "
        f"{len(candidates)} total candidate PIDs."
    )
    logger.info(
        "Signal forwarding candidate scan: "
        f"descendants={len(descendants)}, same_group={len(group_peers)}, "
        f"user_candidates={len(user_candidates)}, "
        f"total={len(candidates)}"
    )

    forwarded = 0
    for pid in sorted(candidates):
        if pid == root_pid:
            continue
        cmdline = _read_cmdline(pid)
        logger.debug(
            f"PID {pid}: cmdline='{cmdline}'"
        )
        # Forward signal to all descendants, including nested snakemake/python.
        # The root jobstep executor already has the handler registered,
        # so we propagate down the entire process tree to reach user code.
        try:
            os.kill(pid, signum)
            logger.debug(f"Sent signal {signum} to PID {pid}.")
            forwarded += 1
        except ProcessLookupError:
            logger.debug(f"PID {pid} already terminated.")
            continue
        except PermissionError as err:
            logger.warning(f"Failed to forward signal {signum} to pid {pid}: {err}")

    return forwarded

def _get_user_process_pids(uid: int) -> set[int]:
    """Return all process IDs for the given uid using ps output."""
    try:
        user_name = pwd.getpwuid(uid).pw_name
    except KeyError:
        return set()

    try:
        result = subprocess.run(
            ["ps", "-u", user_name, "-o", "pid="],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return set()

    if result.returncode != 0:
        return set()

    pids: set[int] = set()
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            pids.add(int(text))
        except ValueError:
            continue

    return pids


def _get_same_process_group_pids(root_pid: int) -> set[int]:
    """Return all PIDs in the same process group as root_pid."""
    root_stat_path = f"/proc/{root_pid}/stat"
    try:
        with open(root_stat_path, "r", encoding="utf-8") as f:
            root_stat = f.read()
    except OSError:
        return set()

    root_close_paren_index = root_stat.rfind(")")
    if root_close_paren_index == -1:
        return set()

    root_fields = root_stat[root_close_paren_index + 2 :].split()
    if len(root_fields) < 3:
        return set()

    try:
        root_pgrp = int(root_fields[2])
    except ValueError:
        return set()

    try:
        proc_entries = os.listdir("/proc")
    except OSError:
        return set()

    same_group: set[int] = set()
    for entry in proc_entries:
        if not entry.isdigit():
            continue

        pid = int(entry)
        stat_path = f"/proc/{pid}/stat"
        try:
            with open(stat_path, "r", encoding="utf-8") as f:
                stat_content = f.read()
        except OSError:
            continue

        close_paren_index = stat_content.rfind(")")
        if close_paren_index == -1:
            continue

        fields = stat_content[close_paren_index + 2 :].split()
        if len(fields) < 3:
            continue

        try:
            pgrp = int(fields[2])
        except ValueError:
            continue

        if pgrp == root_pgrp:
            same_group.add(pid)

    return same_group


def _get_descendant_pids(root_pid: int) -> set[int]:
    """Return all descendant PIDs for a process by scanning /proc."""
    parent_to_children: dict[int, set[int]] = {}

    try:
        proc_entries = os.listdir("/proc")
    except OSError as err:
        return set()

    for entry in proc_entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        stat_path = f"/proc/{pid}/stat"
        try:
            with open(stat_path, "r", encoding="utf-8") as f:
                stat_content = f.read()
        except OSError:
            continue

        close_paren_index = stat_content.rfind(")")
        if close_paren_index == -1:
            continue
        fields = stat_content[close_paren_index + 2 :].split()
        if len(fields) < 2:
            continue
        try:
            ppid = int(fields[1])
        except ValueError:
            continue

        parent_to_children.setdefault(ppid, set()).add(pid)

    descendants: set[int] = set()
    stack = list(parent_to_children.get(root_pid, set()))
    while stack:
        current = stack.pop()
        if current in descendants:
            continue
        descendants.add(current)
        stack.extend(parent_to_children.get(current, set()))

    return descendants


def _read_cmdline(pid: int) -> str:
    """Read process command line from /proc; return empty string on failure."""
    cmdline_path = f"/proc/{pid}/cmdline"
    try:
        with open(cmdline_path, "rb") as f:
            raw = f.read()
    except OSError:
        return ""

    # /proc/<pid>/cmdline is NUL-delimited.
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()


def _is_snakemake_cmdline(cmdline: str) -> bool:
    """Return whether a process command line corresponds to snakemake."""
    return "snakemake" in cmdline.lower()


def _is_python_cmdline(cmdline: str) -> bool:
    """Return whether a process command line corresponds to a Python process."""
    return bool(re.search(r"(?:^|[\s/])python(?:[0-9.]+)?(?:\s|$)", cmdline.lower()))


def _parse_compact_array_execs(raw_array_execs: str) -> dict | None:
    """Parse compact dict-like mapping with bare hex values.

    Expected shape: {2: a0ff..., 3: b19e...}
    """
    text = raw_array_execs.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None

    inner = text[1:-1]
    if not inner.strip():
        return {}

    pair_pattern = re.compile(r"\s*([0-9]+)\s*:\s*([0-9a-fA-F]+)\s*(?:,|$)")
    parsed: dict[str, str] = {}
    position = 0
    while position < len(inner):
        while position < len(inner) and inner[position].isspace():
            position += 1
        if position >= len(inner):
            break

        match = pair_pattern.match(inner, position)
        if match is None:
            return None

        parsed[match.group(1)] = match.group(2)
        position = match.end()

    return parsed
