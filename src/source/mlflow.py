from dataclasses import dataclass
from typing import Iterable, Union, Callable, TypeVar

import datahub.emitter.mce_builder as builder
from datahub.configuration.common import ConfigModel
from datahub.emitter.mcp import MetadataChangeProposalWrapper
from datahub.ingestion.api.common import PipelineContext, WorkUnit
from datahub.ingestion.api.source import Source, SourceReport
from datahub.ingestion.api.workunit import MetadataWorkUnit
from datahub.metadata.schema_classes import (
    _Aspect,
    GlobalTagsClass,
    MLHyperParamClass,
    MLMetricClass,
    MLModelGroupPropertiesClass,
    MLModelPropertiesClass,
    TagAssociationClass,
    TagPropertiesClass,
    VersionTagClass,
)
from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.entities.model_registry import RegisteredModel, ModelVersion
from mlflow.store.entities import PagedList
from pydantic.fields import Field

T = TypeVar('T')


class MLflowConfig(ConfigModel):
    tracking_uri: str = Field(
        default=None,
        description="Tracking server URI",
    )
    registry_uri: str = Field(
        default=None,
        description="Registry server URI",
    )
    model_name_separator: str = Field(
        default="_",
        description="A string which separates mlflow registered model name from its version (e.g. model_1 or model-1)",
    )
    env: str = Field(
        default=builder.DEFAULT_ENV,
        description="Environment to use in namespace when constructing URNs.",
    )
    tracking_ui_address: str = Field(
        default=None,
        description="""
        Tracking UI address if it is differ from the tracking_uri. Starts with http(s):// (e.g. http://localhost:5000)
        """,
    )


@dataclass
class MLflowRegisteredModelStageInfo:
    name: str
    description: str
    color_hex: str


class MLflowSource(Source):
    """This is an MLflow Source"""

    platform = "mlflow"
    registered_model_stages_info = (
        MLflowRegisteredModelStageInfo(
            name="Production",
            description="Production Stage for an ML model in MLflow Model Registry",
            color_hex="#308613",
        ),
        MLflowRegisteredModelStageInfo(
            name="Staging",
            description="Staging Stage for an ML model in MLflow Model Registry",
            color_hex="#FACB66",
        ),
        MLflowRegisteredModelStageInfo(
            name="Archived",
            description="Archived Stage for an ML model in MLflow Model Registry",
            color_hex="#5D7283",
        ),
        MLflowRegisteredModelStageInfo(
            name="None",
            description="None Stage for an ML model in MLflow Model Registry",
            color_hex="#F2F4F5",
        ),
    )

    def __init__(self, ctx: PipelineContext, config: MLflowConfig):
        super().__init__(ctx)
        self.config = config
        self.report = SourceReport()
        self.client = MlflowClient(
            tracking_uri=self.config.tracking_uri,
            registry_uri=self.config.registry_uri,
        )

    def get_report(self) -> SourceReport:
        return self.report

    def get_workunits(self) -> Iterable[WorkUnit]:
        # at first, create tags for each stage in mlflow model registry
        yield from self._get_mlflow_model_registry_stage_workunits()
        # then ingest metadata for model in a model registry and associate their stages with previously created tags
        yield from self._get_ml_model_workunits()

    def _get_mlflow_model_registry_stage_workunits(self) -> Iterable[WorkUnit]:
        for stage_info in self.registered_model_stages_info:
            tag_urn = self._make_stage_tag_urn(stage_info.name)
            tag_properties = TagPropertiesClass(
                name=self._make_stage_tag_name(stage_info.name),
                description=stage_info.description,
                colorHex=stage_info.color_hex,
            )
            wu = self._create_workunit(urn=tag_urn, aspect=tag_properties)
            yield wu

    def _make_stage_tag_urn(self, stage_name: str) -> str:
        tag_name = self._make_stage_tag_name(stage_name)
        tag_urn = builder.make_tag_urn(tag_name)
        return tag_urn

    def _make_stage_tag_name(self, stage_name: str) -> str:
        return f"{self.platform}_{stage_name.lower()}"

    def _create_workunit(self, urn: str, aspect: _Aspect) -> MetadataWorkUnit:
        mcp = MetadataChangeProposalWrapper(entityUrn=urn, aspect=aspect)
        wu = MetadataWorkUnit(id=urn, mcp=mcp)
        self.report.report_workunit(wu)
        return wu

    def _get_ml_model_workunits(self) -> Iterable[WorkUnit]:
        registered_models = self._get_mlflow_registered_models()
        for registered_model in registered_models:
            yield self._get_ml_group_workunit(registered_model)
            model_versions = self._get_mlflow_model_versions(registered_model)
            for model_version in model_versions:
                run = self._get_mlflow_run(model_version)
                yield self._get_ml_model_properties_workunit(
                    registered_model=registered_model,
                    model_version=model_version,
                    run=run,
                )
                yield self._get_global_tags_workunit(model_version=model_version)

    def _get_mlflow_registered_models(self) -> Iterable[RegisteredModel]:
        registered_models = self._traverse_mlflow_search_func(
            search_func=self.client.search_registered_models,
        )
        return registered_models

    @staticmethod
    def _traverse_mlflow_search_func(search_func: Callable[..., PagedList[T]], **kwargs) -> Iterable[T]:
        next_page_token = None
        all_pages_where_traversed = False
        while not all_pages_where_traversed:
            paged_list = search_func(page_token=next_page_token, **kwargs)
            yield from paged_list.to_list()
            next_page_token = paged_list.token
            if not next_page_token:
                all_pages_where_traversed = True

    def _get_ml_group_workunit(self, registered_model: RegisteredModel) -> WorkUnit:
        ml_model_group_urn = self._make_ml_model_group_urn(registered_model)
        ml_model_group_properties = MLModelGroupPropertiesClass(
            customProperties=registered_model.tags,
            description=registered_model.description,
            createdAt=registered_model.creation_timestamp,
        )
        wu = self._create_workunit(urn=ml_model_group_urn, aspect=ml_model_group_properties)
        return wu

    def _make_ml_model_group_urn(self, registered_model: RegisteredModel) -> str:
        urn = builder.make_ml_model_group_urn(
            platform=self.platform,
            group_name=registered_model.name,
            env=self.config.env,
        )
        return urn

    def _get_mlflow_model_versions(self, registered_model: RegisteredModel) -> Iterable[ModelVersion]:
        filter_string = f"name = '{registered_model.name}'"
        model_versions = self._traverse_mlflow_search_func(
            search_func=self.client.search_model_versions,
            filter_string=filter_string,
        )
        return model_versions

    def _get_mlflow_run(self, model_version: ModelVersion) -> Union[None, Run]:
        if model_version.run_id:
            run = self.client.get_run(model_version.run_id)
            return run
        else:
            return None

    def _get_ml_model_properties_workunit(
            self,
            registered_model: RegisteredModel,
            model_version: ModelVersion,
            run: Union[None, Run],
    ) -> WorkUnit:
        # we use mlflow registered model as a datahub ml model group
        ml_model_group_urn = self._make_ml_model_group_urn(registered_model)
        # we use each mlflow registered model version as a datahub ml model
        ml_model_urn = self._make_ml_model_urn(model_version)
        # if a model was registered without an associated run then hyperparams and metrics are not available
        if run:
            hyperparams = [MLHyperParamClass(name=k, value=str(v)) for k, v in run.data.params.items()]
            training_metrics = [MLMetricClass(name=k, value=str(v)) for k, v in run.data.metrics.items()]
        else:
            hyperparams = None
            training_metrics = None
        ml_model_properties = MLModelPropertiesClass(
            customProperties=model_version.tags,
            externalUrl=self._make_external_url(model_version),
            description=model_version.description,
            date=model_version.creation_timestamp,
            version=VersionTagClass(versionTag=str(model_version.version)),
            hyperParams=hyperparams,
            trainingMetrics=training_metrics,
            # mlflow tags are dicts, but datahub tags are lists. currently use only keys from mlflow tags
            tags=list(model_version.tags.keys()),
            groups=[ml_model_group_urn],
        )
        wu = self._create_workunit(urn=ml_model_urn, aspect=ml_model_properties)
        return wu

    def _make_ml_model_urn(self, model_version: ModelVersion) -> str:
        urn = builder.make_ml_model_urn(
            platform=self.platform,
            model_name=f"{model_version.name}{self.config.model_name_separator}{model_version.version}",
            env=self.config.env,
        )
        return urn

    def _make_external_url(self, model_version: ModelVersion) -> Union[None, str]:
        if self.config.tracking_ui_address:
            base_uri = self.config.tracking_ui_address
        else:
            base_uri = self.client.tracking_uri
        if base_uri.startswith("http"):
            return f"{base_uri.rstrip('/')}/#/models/{model_version.name}/versions/{model_version.version}"
        else:
            return None

    def _get_global_tags_workunit(self, model_version: ModelVersion) -> WorkUnit:
        global_tags = GlobalTagsClass(
            tags=[
                TagAssociationClass(tag=self._make_stage_tag_urn(model_version.current_stage)),
            ]
        )
        wu = self._create_workunit(urn=self._make_ml_model_urn(model_version), aspect=global_tags)
        return wu

    @classmethod
    def create(cls, config_dict: dict, ctx: PipelineContext) -> Source:
        config = MLflowConfig.parse_obj(config_dict)
        return cls(ctx, config)
