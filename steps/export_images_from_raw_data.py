import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import click
import mlflow
from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets
from dotenv import load_dotenv

from tools.mnist_decoder import MnistDecoder


def launch(shell_out_cmd: str) -> None:
    args = shlex.split(shell_out_cmd)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in proc.stdout:
        sys.stdout.buffer.write(line)
    for line in proc.stderr:
        sys.stdout.buffer.write(line)


@click.command()
@click.option("--input-dir", default="data/inbound", type=str)
@click.option("--output-dir", default="data/etl", type=str)
@click.option("--cache", default=True, type=bool)
def export_images_from_raw_data(input_dir: str, output_dir: str, cache: bool):
    with mlflow.start_run(nested=True) as mlrun:

        if cache and Path(output_dir).exists():
            return

        if not cache and Path(output_dir).exists():
            shutil.rmtree(path=output_dir)

        decoder: MnistDecoder = MnistDecoder(input_path=Path(input_dir), output_path=Path(output_dir))
        decoder.export_data()


if __name__ == "__main__":
    # load defined environmental variables
    load_ae5_user_secrets()
    load_dotenv(dotenv_path="env/env.dev")

    export_images_from_raw_data()
