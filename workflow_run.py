import shlex
import subprocess
import sys

from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets
from dotenv import load_dotenv


def launch(shell_out_cmd: str) -> None:
    args = shlex.split(shell_out_cmd)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in proc.stdout:
        sys.stdout.buffer.write(line)
    for line in proc.stderr:
        sys.stdout.buffer.write(line)


if __name__ == "__main__":
    # load defined environmental variables
    load_ae5_user_secrets()
    load_dotenv(dotenv_path="env/env.ae5.dev")

    cmd: str = "mlflow run . --env-manager local"
    launch(shell_out_cmd=cmd)
