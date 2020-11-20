# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import os
import subprocess


_log = logging.getLogger("util.run")


def run(*args, cwd=None, capture=False, shell=False, env=None, check=True,
        quiet=False):
    """Runs a subprocess.

    Args:
        *args: The subprocess arguments.
        cwd: The working directory. If None, specifies the current directory.
        capture: Whether to capture stdout and stderr.
        shell: Whether to run using the shell.
        env: The environment variables as a dict. If None, inherits the current
            environment.
        check: Whether to raise an error if the return code is not zero.
        quiet: If true, do not print output from the subprocess.

    Returns:
        A subprocess.CompletedProcess instance.
    """
    cmd = [*args]

    _log.info("Running subprocess in '{0}'\n{1}".format(
        cwd or os.getcwd(), cmd))

    output = \
        subprocess.PIPE if capture else (subprocess.DEVNULL if quiet else None)
    completed_process = subprocess.run(
        cmd, cwd=cwd, check=check, stdout=output, stderr=output, env=env,
        shell=shell)

    _log.debug("Subprocess completed. Return code: {}".format(
        completed_process.returncode))

    return completed_process
