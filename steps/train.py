import shlex
import subprocess
import sys

import click
import mlflow
from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets
from dotenv import load_dotenv

from steps.training.trainer import Trainer


def launch(shell_out_cmd: str) -> None:
    args = shlex.split(shell_out_cmd)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in proc.stdout:
        sys.stdout.buffer.write(line)
    for line in proc.stderr:
        sys.stdout.buffer.write(line)


@click.command()
@click.option("--input-dir", default="data/etl/train", type=str)
@click.option("--epochs", default=1, type=int)
@click.option("--early-stop-patience", default=10, type=int)
def train(input_dir: str, epochs: int, early_stop_patience: int):
    with mlflow.start_run(nested=True) as mlrun:
        Trainer().execute(data_dir=input_dir, epochs=epochs, early_stop_patience=early_stop_patience)


if __name__ == "__main__":
    # load defined environmental variables
    load_ae5_user_secrets()
    load_dotenv(dotenv_path="env/env.dev")

    train()
