import datetime

import pytest
from datahub.ingestion.api.common import PipelineContext
from mlflow.entities.model_registry.model_version import ModelVersion

from source.mlflow import MLflowSource, MLflowConfig


@pytest.fixture
def tracking_uri(tmp_path) -> str:
    return str(tmp_path / "mlruns")


def test_stages(tracking_uri):
    mlflow_registered_model_stages = {
        "Production",
        "Staging",
        "Archived",
        None,
    }
    source = MLflowSource(
        ctx=PipelineContext(run_id="mlflow-source-test"),
        config=MLflowConfig(tracking_uri=tracking_uri),
    )

    workunits = source._get_mlflow_model_registry_stage_workunits()
    names = [wu.get_metadata()["metadata"].aspect.name for wu in workunits]

    assert len(names) == len(mlflow_registered_model_stages)
    assert set(names) == {"mlflow_" + str(stage).lower() for stage in mlflow_registered_model_stages}


def test_separator(tracking_uri):
    model_name = "abc"
    version = "1"
    name_version_sep = "+"
    source = MLflowSource(
        ctx=PipelineContext(run_id="mlflow-source-test"),
        config=MLflowConfig(
            tracking_uri=tracking_uri,
            model_name_separator=name_version_sep,
        ),
    )
    expected_urn = \
        f"urn:li:mlModel:(urn:li:dataPlatform:mlflow,{model_name}{name_version_sep}{version},{source.config.env})"
    model_version = ModelVersion(
        name=model_name,
        version=version,
        creation_timestamp=datetime.datetime.now(),
    )

    urn = source._make_ml_model_urn(model_version)

    assert urn == expected_urn
