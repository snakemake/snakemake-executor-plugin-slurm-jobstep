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


def should_stage_in(inputfile):
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
    try:
        expanded = subprocess.run(
            ["scontrol", "show", "hostname", "$SLURM_NODELIST"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        raise WorkflowError(
            "Failed to expand SLURM nodelist via `scontrol show hostname`."
        ) from err

    return [host for host in expanded.stdout.splitlines() if host]


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
