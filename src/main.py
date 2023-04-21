from source.mlflow import MLflowSource, MLflowConfig
from datahub.ingestion.api.common import PipelineContext


if __name__ == '__main__':
    source = MLflowSource(
        ctx=PipelineContext(run_id="dummy"),
        config=MLflowConfig(tracking_uri="/Users/grigory/Prog/opensource/datahub-mlflow-source/mlruns"),
    )
    items = list(source._get_mlflow_data())
    print(items)
    # wus = list(source.get_workunits())
    # print(wus)
