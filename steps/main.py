import mlflow
from anaconda.enterprise.mlops.sdk import build_mlflow_client
from anaconda.enterprise.server.common.sdk import load_ae5_user_secrets
from dotenv import load_dotenv


def execute_step(entry_point: str, parameters: dict):
    print(f"Launching new run for entrypoint={entry_point} and parameters={parameters}")
    submitted_run = mlflow.run(uri=".", entry_point=entry_point, parameters=parameters, env_manager="local")
    return build_mlflow_client().get_run(submitted_run.run_id)


def workflow():
    with mlflow.start_run() as active_workflow_run:
        # download data
        download_parameters: dict = {"output-dir": "data/inbound", "cache": True}
        execute_step(entry_point="download_raw_data", parameters=download_parameters)

        # process data
        etl_parameters: dict = {"input-dir": "data/inbound", "output-dir": "data/etl", "cache": True}
        execute_step(entry_point="export_images_from_raw_data", parameters=etl_parameters)

        # train
        train_parameters: dict = {"input-dir": "data/etl/train"}
        execute_step(entry_point="train", parameters=train_parameters)

        # evaluate


if __name__ == "__main__":
    # load defined environmental variables
    load_ae5_user_secrets()
    load_dotenv(dotenv_path="env/env.dev")

    # Launch workflow
    workflow()
