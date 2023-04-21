from mlflow import MlflowClient

if __name__ == '__main__':
    client = MlflowClient(
        tracking_uri="/Users/grigory/Prog/opensource/datahub-mlflow-source/mlruns",
    )

    stage_map = {
        'Staging': 'STG',
        'Production': 'PROD',
    }
    stages_to_use = [
        "Staging",
        "Production",
    ]
    for rm in client.search_registered_models():
        print(rm.name)
        mvs = client.get_latest_versions(
            name=rm.name,
            stages=stages_to_use,
        )
        for mv in mvs:
            print(dict(mv))
            print(stage_map[mv.current_stage])
