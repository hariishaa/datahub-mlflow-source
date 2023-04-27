import datetime
from typing import TypeVar, Union

import pytest
from datahub.ingestion.api.common import PipelineContext
from mlflow import MlflowClient
from mlflow.entities.model_registry import RegisteredModel
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.store.entities import PagedList

from source.mlflow import MLflowSource, MLflowConfig

T = TypeVar('T')


@pytest.fixture
def source() -> MLflowSource:
    return MLflowSource(
        ctx=PipelineContext(run_id="mlflow-source-test"),
        config=MLflowConfig(),
    )


@pytest.fixture
def registered_model(source: MLflowSource) -> RegisteredModel:
    model_name = "abc"
    return RegisteredModel(name=model_name)


@pytest.fixture
def model_version(source: MLflowSource, registered_model: RegisteredModel) -> ModelVersion:
    version = "1"
    return ModelVersion(
        name=registered_model.name,
        version=version,
        creation_timestamp=datetime.datetime.now(),
    )


def dummy_search_func(page_token: Union[None, str], **kwargs) -> PagedList[T]:
    dummy_pages = dict(
        page_1=PagedList(items=["a", "b"], token="page_2"),
        page_2=PagedList(items=["c", "d"], token="page_3"),
        page_3=PagedList(items=["e"], token=None),
    )
    if page_token is None:
        page_to_return = dummy_pages["page_1"]
    else:
        page_to_return = dummy_pages[page_token]
    if kwargs.get("case", "") == "upper":
        page_to_return = PagedList(
            items=[e.upper() for e in page_to_return.to_list()],
            token=page_to_return.token,
        )
    return page_to_return


def test_stages(source):
    mlflow_registered_model_stages = {
        "Production",
        "Staging",
        "Archived",
        None,
    }
    workunits = source._get_mlflow_model_registry_stage_workunits()
    names = [wu.get_metadata()["metadata"].aspect.name for wu in workunits]

    assert len(names) == len(mlflow_registered_model_stages)
    assert set(names) == {"mlflow_" + str(stage).lower() for stage in mlflow_registered_model_stages}


def test_config_model_name_separator(source, model_version):
    name_version_sep = "+"
    source.config.model_name_separator = name_version_sep
    expected_model_name = f"{model_version.name}{name_version_sep}{model_version.version}"
    expected_urn = f"urn:li:mlModel:(urn:li:dataPlatform:mlflow,{expected_model_name},{source.config.env})"

    urn = source._make_ml_model_urn(model_version)

    assert urn == expected_urn


def test_model_without_run(source, registered_model, model_version):
    run = source._get_mlflow_run(model_version)
    wu = source._get_ml_model_properties_workunit(
        registered_model=registered_model,
        model_version=model_version,
        run=run,
    )
    aspect = wu.get_metadata()["metadata"].aspect

    assert aspect.hyperParams is None
    assert aspect.trainingMetrics is None


def test_traverse_mlflow_search_func(source):
    expected_items = ["a", "b", "c", "d", "e"]

    items = list(source._traverse_mlflow_search_func(dummy_search_func))

    assert items == expected_items


def test_traverse_mlflow_search_func_with_kwargs(source):
    expected_items = ["A", "B", "C", "D", "E"]

    items = list(source._traverse_mlflow_search_func(dummy_search_func, case="upper"))

    assert items == expected_items


def test_make_external_link_local(source, model_version):
    expected_url = None

    url = source._make_external_url(model_version)

    assert url == expected_url


def test_make_external_link_remote(source, model_version):
    tracking_uri_remote = "https://dummy-mlflow-tracking-server.org"
    source.client = MlflowClient(tracking_uri=tracking_uri_remote)
    expected_url = f"{tracking_uri_remote}/#/models/{model_version.name}/versions/{model_version.version}"

    url = source._make_external_url(model_version)

    assert url == expected_url


def test_make_external_link_tracking_ui_address(source, model_version):
    tracking_ui_address = "http://localhost:5000"
    source.config.tracking_ui_address = tracking_ui_address
    expected_url = f"{tracking_ui_address}/#/models/{model_version.name}/versions/{model_version.version}"

    url = source._make_external_url(model_version)

    assert url == expected_url
