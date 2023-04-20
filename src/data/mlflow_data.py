from mlflow import MlflowClient

if __name__ == '__main__':
    client = MlflowClient(
        tracking_uri="/Users/grigory/Prog/opensource/datahub-mlflow-source/mlruns",
    )
    client.create_registered_model(
        name="sk-learn-random-forest-reg-model",
        tags=dict(
            model_id=1,
            model_env="test",
        ),
        description="This is the first registered model",
    )

    for rm in client.search_registered_models():
        print(dict(rm))
