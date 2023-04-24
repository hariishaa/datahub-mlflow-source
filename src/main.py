from source.mlflow import MLflowSource, MLflowConfig
from datahub.ingestion.api.common import PipelineContext


if __name__ == '__main__':
    source = MLflowSource(
        ctx=PipelineContext(run_id="dummy"),
        config=MLflowConfig(tracking_uri="/Users/grigory/Prog/opensource/datahub-mlflow-source/mlruns"),
    )
    registered_models = source._get_mlflow_registered_models()
    for rm in registered_models:
        print(f"--- {rm.name} ---")
        model_versions = source._get_mlflow_model_versions(registered_model=rm)
        for mv in model_versions:
            print(f"--- Version {mv.version} ---")
            print(mv)
            run = source._get_mlflow_run(mv)
            if not run:
                continue
            print(run)
            print(run.data.params)
            print(run.data.metrics)

