from mlflow import MlflowClient

if __name__ == '__main__':
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
    )
    client.transition_model_version_stage(
        name=model_name,
        version="1",
        stage="Staging"
    )

    client.create_model_version(
        name=model_name,
        source="mlruns/0/d16076a3ec534311817565e6527539c0/artifacts/sklearn-model",
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
