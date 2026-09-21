import os
import re
from pathlib import Path
import subprocess

from snakemake.io.flags.access_patterns import (
    AccessPattern,
    STORE_KEY,
)
from snakemake_interface_common.exceptions import WorkflowError


def is_ondemand_eligible(inputfile):
    """
    Determine whether an input file should be staged in based on its access pattern.
    """
    pattern = inputfile.flags.get(STORE_KEY)
    return pattern in {AccessPattern.RANDOM, AccessPattern.MULTI}


def get_file_size(inputfile):
    """
    Get the size of the input file if available, otherwise return None.
    """
    return os.path.getsize(inputfile)


def get_nodelist():
    """
    Get the list of nodes allocated for the job from SLURM environment variables
    """
    evaluate_nodelist = os.environ.get("SLURM_NODELIST")
    if not evaluate_nodelist:
        raise WorkflowError("unable to get 'SLURM_NODELIST'")
    try:
        expanded = subprocess.run(
            ["scontrol", "show", "hostname", evaluate_nodelist],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        raise WorkflowError(
            "Failed to expand SLURM nodelist via `scontrol show hostname`."
        ) from err

    hosts = [host for host in expanded.stdout.splitlines() if host]
    if not hosts:
        raise WorkflowError(
            "Failed to expand SLURM nodelist via `scontrol show hostname`: "
            "no nodes were returned."
        )

    return hosts


def check_filesystem_availability(remote_directory):
    """
    Check the available size on the filesystem where the remote directory is located.
    """
    try:
        statvfs = os.statvfs(remote_directory)
        return statvfs.f_bavail * statvfs.f_frsize
    except OSError as err:
        raise WorkflowError(
            f"Failed to check filesystem for remote directory {remote_directory}."
        ) from err


def ensure_stage_in_directory(remote_directory):
    """
    Ensure that the stage-in directory exists on the remote hosts.
    If it does not exist, the job will fail.
    """
    nodelist = get_nodelist()
    directory = Path(remote_directory)

    for node in nodelist:
        if node == get_nodename():
            # if the node is the same as the current node, we can just
            # check the directory locally
            if not directory.exists():
                raise WorkflowError(
                    f"Failed to find stage-in directory {remote_directory} on {node}."
                )
            continue
        try:
            subprocess.run(
                ["ssh", node, "ls", directory.as_posix()],
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as err:
            raise WorkflowError(
                f"Failed to find stage-in directory {remote_directory} on {node}."
            ) from err


def get_file_system_size(path):
    """
    Get the available size of the filesystem where the path is located in GB.
    """
    try:
        statvfs = os.statvfs(path)
        available_bytes = statvfs.f_bavail * statvfs.f_frsize
        return round(available_bytes / (1024**3), 2)
    except OSError as err:
        raise WorkflowError(f"Failed to check filesystem for path {path}.") from err


def stage_in_sbcast(inpath, remote_directory):
    """
    `sbcast` is a SLURM built-in utility for staging files to the compute nodes.
    It works on single files and is best for files smaller 2GB.
    It's signature is `sbcast <local_path> <remote_path>`,
    where the remote path is expected to be on a shared filesystem,
    but can be outside of the job's working directory.
    The utility takes care of staging the file to the compute nodes and
    placing it at the specified remote path.

    Note: it expects a full absolute input path and a full absolute remote path
    """
    # The remote path is combined from the input file name and the
    # remote directory.

    fname = Path(inpath).name
    remote_path = Path(remote_directory) / fname
    try:
        subprocess.run(
            ["sbcast", inpath, str(remote_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        raise WorkflowError(
            f"Failed to stage in file {inpath} via sbcast to {remote_path}."
        ) from err

    try:
        staged_path = inpath.__class__(
            str(remote_path), rule=getattr(inpath, "rule", None)
        )
        if hasattr(staged_path, "clone_flags"):
            staged_path.clone_flags(inpath)
        return staged_path
    except Exception:
        return str(remote_path)


def get_nodename():
    """
    Get the name of the current node from SLURM environment variables.
    """
    nodename = os.environ.get("SLURMD_NODENAME")
    if not nodename:
        raise WorkflowError(
            "Failed to get current node name from SLURM environment "
            "variable SLURMD_NODENAME."
        )
    return nodename


def stage_in_scp(inpath, remote_directory):
    """
    `scp` is a standard utility for copying files over SSH. It can be used for
    staging-in files. For `scp` to work, host based login via SSH or passphrase
    based login must be set up between the submit host and the
    compute nodes, and the remote path must be on a shared filesystem.
    """
    fname = Path(inpath).name
    remote_path = Path(remote_directory) / fname

    nodelist = get_nodelist()

    # we need to iterate over the nodelist and scp to each node,
    # as scp does not have a built-in way to copy to multiple hosts
    for node in nodelist:
        if node == get_nodename():
            # if the node is the same as the current node, we can just
            # copy the file locally
            try:
                subprocess.run(
                    ["cp", inpath, str(remote_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as err:
                raise WorkflowError(
                    f"Failed to stage in file {inpath} via local copy to {remote_path}."
                ) from err
            continue
        else:
            try:
                subprocess.run(
                    ["scp", inpath, f"{node}:{remote_path}"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as err:
                raise WorkflowError(
                    f"Failed to stage in file {inpath} via scp to {remote_path}."
                ) from err

    try:
        staged_path = inpath.__class__(
            str(remote_path), rule=getattr(inpath, "rule", None)
        )
        if hasattr(staged_path, "clone_flags"):
            staged_path.clone_flags(inpath)
        return staged_path
    except Exception:
        return str(remote_path)
