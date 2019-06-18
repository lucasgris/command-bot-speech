import subprocess
import sys


def system(command: str, debug: bool = False):
    """
    Runs a system command

    debug: bool
        If True, will print the command.
    return_out: bool
    """
    if debug:
        print(command)
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               shell=True)
    stdout, stderr = process.communicate()
    if debug and stderr is not None:
        print(stderr.decode(sys.stdout.encoding))
    return stdout.decode(sys.stdout.encoding)
