import os
import re
from pathlib import Path
import subprocess

from snakemake.io.flags.access_patterns import (
    AccessPattern,
    STORE_KEY,
)
from snakemake_interface_common.exceptions import WorkflowError

_ENV_MARKER = re.compile(r"__ENV(?:__|_)([A-Z0-9]+(?:_[A-Z0-9]+)*)__")


def expand_node_local_prefix(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), "")

    return _ENV_MARKER.sub(repl, value)


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
    size = os.path.getsize(inputfile)
    # return file size in GB - we do not care for the exact value
    return size // (1024**3)


def get_nodelist():
    """
    Get the list of nodes allocated for the job from SLURM environment variables
    """
    evaluate_nodelist = os.environ.get("SLURM_NODELIST")
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

    return [host for host in expanded.stdout.splitlines() if host]


def check_filesystem_availability(remote_directory):
    """
    Check the available size on the filesystem where the remote directory is located.
    """
    try:
        statvfs = os.statvfs(remote_directory)
        # Calculate available space in GB
        available_gb = (statvfs.f_bavail * statvfs.f_frsize) // (1024**3)
        return available_gb
    except OSError as err:
        raise WorkflowError(
            f"Failed to check filesystem for remote directory {remote_directory}."
        ) from err


def stage_in_sbcast(inpath, remote_directory):
    """
    `sbcast` is a SLURM-build-in utitlity for staging files to the compute nodes.
    It works on single files and best of files smaller 2GB.
    It's signature is `sbcast <local_path> <remote_path>`,
    where the remote path is expected to be on a shared filesystem,
    but can be outside of the job's working directory.
    The utility takes care of staging the file to the compute nodes and
    placing it at the specified remote path.

    Note: it expexts a full absolute input path and a full absolute remote path
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
    if nodename is None:
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
