import os
from random import random, randint

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor


def generate_registered_model():
    client = MlflowClient(
        tracking_uri="/Users/grigory/Prog/opensource/datahub-mlflow-source/mlruns",
    )
    model_name = "sk-learn-random-forest-reg-model"

    client.delete_registered_model(name=model_name)

    client.create_registered_model(
        name=model_name,
        tags=dict(
            model_id=1,
            model_env="test",
        ),
        description="This is the first registered model",
    )

    client.create_model_version(
        name=model_name,
        source="mlruns/0/d16076a3ec534311817565e6527539c0/artifacts/sklearn-model",
        tags=dict(model_version_id=1),
    )
    client.transition_model_version_stage(
        name=model_name,
        version="1",
        stage="Staging"
    )

    client.create_model_version(
        name=model_name,
        source="mlruns/0/d16076a3ec534311817565e6527539c0/artifacts/sklearn-model",
        tags=dict(model_version_id=2),
    )
    client.transition_model_version_stage(
        name=model_name,
        version="2",
        stage="Archived"
    )

    client.create_model_version(
        name=model_name,
        source="mlruns/0/d16076a3ec534311817565e6527539c0/artifacts/sklearn-model",
    )
    client.transition_model_version_stage(
        name=model_name,
        version="3",
        stage="Production"
    )

    client.create_model_version(
        name=model_name,
        source="mlruns/0/d16076a3ec534311817565e6527539c0/artifacts/sklearn-model",
    )


def generate_registered_model_from_runs():
    active_run = mlflow.start_run(
        run_name="my_first_run",
        tags=dict(
            run_purpose="POC",
            run_creator="hariishaa",
        ),
        description="This run was created for testing purposes",
    )
    with active_run:
        params = {"n_estimators": 5, "random_state": 42}
        sk_learn_rfr = RandomForestRegressor(**params)

        # Log parameters and metrics using the MLflow APIs
        # mlflow.log_params(params)
        # mlflow.log_param("param_1", randint(0, 100))
        mlflow.log_metrics({"metric_1": random(), "metric_2": random() + 1})

        # Log the sklearn model and register as version 1
        mlflow.sklearn.log_model(
            sk_model=sk_learn_rfr,
            artifact_path="sklearn-model",
            registered_model_name="sk-learn-random-forest-reg-model-run-1",
        )


def generate_nested_runs():
    # Create nested runs
    experiment_id = mlflow.create_experiment("experiment1")
    with mlflow.start_run(
            run_name="PARENT_RUN",
            experiment_id=experiment_id,
            tags={"version": "v1", "priority": "P1"},
            description="parent",
    ) as parent_run:
        mlflow.log_param("parent", "yes")
        with mlflow.start_run(
                run_name="CHILD_RUN",
                experiment_id=experiment_id,
                description="child",
                nested=True,
        ) as child_run:
            mlflow.log_param("child", "yes")

    print("parent run:")

    print("run_id: {}".format(parent_run.info.run_id))
    print("description: {}".format(parent_run.data.tags.get("mlflow.note.content")))
    print("version tag value: {}".format(parent_run.data.tags.get("version")))
    print("priority tag value: {}".format(parent_run.data.tags.get("priority")))
    print("--")

    # Search all child runs with a parent id
    query = "tags.mlflow.parentRunId = '{}'".format(parent_run.info.run_id)
    results = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=query)
    print("child runs:")
    print(results[["run_id", "params.child", "tags.mlflow.runName"]])


def generate_registered_model_with_run():
    client = MlflowClient(
        tracking_uri="/Users/grigory/Prog/opensource/datahub-mlflow-source/mlruns",
    )
    experiment_name = "test-experiment"
    run_name = "test-run"
    model_name = "test-model"
    test_experiment_id = client.create_experiment(experiment_name)
    test_run = client.create_run(
        experiment_id=test_experiment_id,
        run_name=run_name,
    )
    client.log_param(
        run_id=test_run.info.run_id,
        key="p",
        value=1,
    )
    client.log_metric(
        run_id=test_run.info.run_id,
        key="m",
        value=0.85,
    )
    client.create_registered_model(
        name=model_name,
        tags=dict(
            model_id=1,
            model_env="test",
        ),
        description="This is the first registered model",
    )
    client.create_model_version(
        name=model_name,
        source="dummy_dir/dummy_file",
        run_id=test_run.info.run_id,
        tags=dict(model_version_id=1),
    )
    client.transition_model_version_stage(
        name=model_name,
        version="1",
        stage="Archived",
    )


if __name__ == '__main__':
    tracking_uri = "/Users/grigory/Prog/opensource/datahub-mlflow-source/mlruns"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.doctor()

    # print("generate_registered_model()")
    # generate_registered_model()
    # print("generate_registered_model_from_runs()")
    # generate_registered_model_from_runs()
    # print("generate_nested_runs()")
    # generate_nested_runs()
    print("generate_registered_model_with_run()")
    generate_registered_model_with_run()
