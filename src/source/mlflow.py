from dataclasses import dataclass
from typing import Iterable, List, Union, Callable, Any, T

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


class MLflowConfig(ConfigModel):
    tracking_uri: str = Field(
        default=None,
        # https://mlflow.org/docs/latest/python_api/mlflow.html?highlight=registry_uri#mlflow.set_tracking_uri
        description="Tracking server URI",
    )
    registry_uri: str = Field(
        default=None,
        # https://mlflow.org/docs/latest/python_api/mlflow.html?highlight=registry_uri#mlflow.set_registry_uri
        description="Registry server URI",
    )
    env: str = Field(
        default=builder.DEFAULT_ENV,
        description="Environment to use in namespace when constructing URNs.",
    )


@dataclass
class MLflowRegisteredModelStageInfo:
    name: str
    description: str
    color_hex: str


# todo:
# Сделать параметризированный разделитель для названия моделей (_/-)
# Проверить окончание запятыми во всех () и []
# Попробовать поставить mlflow-skinny поверх mlflow и заменить mlflow mlflow-skinny
class MLflowSource(Source):
    # todo: make it better
    """This is an MLflow Source"""

    platform = "mlflow"
    # todo: make immutable
    registered_model_stages_info = [
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
    ]

    def __init__(self, ctx: PipelineContext, config: MLflowConfig):
        super().__init__(ctx)
        self.config = config
        # todo: make custom report?
        self.report = SourceReport()
        self.client = MlflowClient(
            tracking_uri=self.config.tracking_uri,
            registry_uri=self.config.registry_uri,
        )
        self.env = self.config.env

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
            wu = self._create_workunit(
                urn=tag_urn,
                aspect=tag_properties,
            )
            yield wu

    def _make_stage_tag_urn(self, stage_name: str) -> str:
        tag_name = self._make_stage_tag_name(stage_name)
        tag_urn = builder.make_tag_urn(tag_name)
        return tag_urn

    def _make_stage_tag_name(self, stage_name: str) -> str:
        return f"{self.platform}_{stage_name.lower()}"

    def _create_workunit(self, urn: str, aspect: _Aspect) -> MetadataWorkUnit:
        mcp = MetadataChangeProposalWrapper(
            entityUrn=urn,
            aspect=aspect,
        )
        wu = MetadataWorkUnit(
            id=urn,
            mcp=mcp,
        )
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
                # todo: make oneliner
                yield self._get_global_tags_workunit(
                    model_version=model_version,
                )

    # todo: remove max_results?
    # max_results here is for debugging purposes
    def _get_mlflow_registered_models(self, max_results: int = 1) -> Iterable[RegisteredModel]:
        registered_models = self._traverse_mlflow_search_func(
            search_func=self.client.search_registered_models,
            max_results=max_results,
        )
        return registered_models

    @staticmethod
    def _traverse_mlflow_search_func(search_func: Callable[..., PagedList[T]], **kwargs) -> Iterable[T]:
        next_page_token = None
        all_pages_where_traversed = False
        while not all_pages_where_traversed:
            paged_list = search_func(
                **kwargs,
                page_token=next_page_token,
            )
            yield from paged_list.to_list()
            next_page_token = paged_list.token
            if not next_page_token:
                all_pages_where_traversed = True

    def _get_ml_group_workunit(self, registered_model: RegisteredModel) -> WorkUnit:
        ml_model_group_urn = self._make_ml_model_group_urn(registered_model)
        # todo: add other options?
        # version
        ml_model_group_properties = MLModelGroupPropertiesClass(
            customProperties=registered_model.tags,
            description=registered_model.description,
            createdAt=registered_model.creation_timestamp,
        )
        wu = self._create_workunit(
            urn=ml_model_group_urn,
            aspect=ml_model_group_properties,
        )
        return wu

    def _make_ml_model_group_urn(self, registered_model: RegisteredModel) -> str:
        urn = builder.make_ml_model_group_urn(
            platform=self.platform,
            group_name=registered_model.name,
            env=self.env,
        )
        return urn

    # todo: remove max_results?
    # max_results here is for debugging purposes
    def _get_mlflow_model_versions(
            self,
            registered_model: RegisteredModel,
            max_results: int = 10000
    ) -> Iterable[ModelVersion]:
        filter_string = f"name = '{registered_model.name}'"
        model_versions = self._traverse_mlflow_search_func(
            search_func=self.client.search_model_versions,
            filter_string=filter_string,
            max_results=max_results,
        )
        return model_versions

    def _get_mlflow_run(self, model_version: ModelVersion) -> Union[Run, None]:
        if model_version.run_id:
            run = self.client.get_run(model_version.run_id)
            return run
        else:
            return None

    def _get_ml_model_properties_workunit(
            self,
            registered_model: RegisteredModel,
            model_version: ModelVersion,
            run: Union[Run, None]
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
            externalUrl=model_version.run_link,
            description=model_version.description,
            date=model_version.creation_timestamp,
            version=VersionTagClass(versionTag=str(model_version.version)),
            hyperParams=hyperparams,
            trainingMetrics=training_metrics,
            # mlflow tags are dicts, but datahub tags are lists. currently use only keys from mlflow tags
            tags=list(model_version.tags),
            groups=[ml_model_group_urn],
        )
        wu = self._create_workunit(
            urn=ml_model_urn,
            aspect=ml_model_properties,
        )
        return wu

    def _make_ml_model_urn(self, model_version: ModelVersion) -> str:
        urn = builder.make_ml_model_urn(
            platform=self.platform,
            model_name=f"{model_version.name}_{model_version.version}",
            env=self.env,
        )
        return urn

    def _get_global_tags_workunit(self, model_version: ModelVersion) -> WorkUnit:
        global_tags = GlobalTagsClass(
            tags=[
                TagAssociationClass(tag=self._make_stage_tag_urn(model_version.current_stage))
            ]
        )
        wu = self._create_workunit(
            urn=self._make_ml_model_urn(model_version),
            aspect=global_tags,
        )
        return wu

    @classmethod
    def create(cls, config_dict: dict, ctx: PipelineContext) -> Source:
        config = MLflowConfig.parse_obj(config_dict)
        return cls(ctx, config)
