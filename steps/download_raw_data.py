"""
Downloads the MovieLens dataset and saves it as an artifact
"""
import shlex
import shutil
import subprocess
from pathlib import Path

import click
from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets
from dotenv import load_dotenv


def shell_out(shell_out_cmd: str, cwd: str) -> tuple[str, str, int]:
    args = shlex.split(shell_out_cmd)
    proc = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        outs, errs = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()
    return outs.decode(encoding="utf-8"), errs.decode(encoding="utf-8"), proc.returncode


@click.command()
@click.option("--output-dir", default="data/inbound", type=str)
@click.option("--cache", default=True, type=bool)
def download_raw_data(output_dir: str, cache: bool):
    # Cache controls
    # if False, we dont use a cache (clean, and redownload), otherwise use existing structures and download if missing.

    if cache and Path(output_dir).exists():
        return

    if not cache and Path(output_dir).exists():
        shutil.rmtree(path=output_dir)

    _download(output_dir=output_dir)


def _download(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd: str = "wget --recursive --level=1 --cut-dirs=3 --no-host-directories --accept '*.gz' http://yann.lecun.com/exdb/mnist/"
    shell_out(shell_out_cmd=cmd, cwd=output_dir)


if __name__ == "__main__":
    # load defined environmental variables
    load_ae5_user_secrets()
    load_dotenv(dotenv_path="env/.env.ae5.dev")

    download_raw_data()
