import shlex
import subprocess
import sys

import click
import mlflow
from dotenv import load_dotenv
from src.anaconda.ae5.secrets import load_ae5_user_secrets

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
def train(input_dir: str):
    with mlflow.start_run(nested=True) as mlrun:
        Trainer().execute(input_dir)


if __name__ == "__main__":
    # load defined environmental variables
    load_dotenv(dotenv_path="env/.env.ae5.dev")
    load_ae5_user_secrets()

    train()
