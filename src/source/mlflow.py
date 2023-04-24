from typing import Iterable, List, Union

import datahub.emitter.mce_builder as builder
from datahub.configuration.common import ConfigModel
from datahub.emitter.mcp import MetadataChangeProposalWrapper
from datahub.ingestion.api.common import PipelineContext, WorkUnit
from datahub.ingestion.api.source import Source, SourceReport
from datahub.ingestion.api.workunit import MetadataWorkUnit
from datahub.metadata.schema_classes import (
    _Aspect,
    MLHyperParamClass,
    MLMetricClass,
    MLModelGroupPropertiesClass,
    MLModelPropertiesClass,
    VersionTagClass,
)
from mlflow import MlflowClient
from mlflow.entities import Run, RunData
from mlflow.entities.model_registry import RegisteredModel, ModelVersion
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


# todo:
# Добавить 4 тега под статусы моделей
# Привязать теги статусов к моделям
# Как-то реализовать ссылку на гуи для моделей
# Проверить окончание запятыми во всех () и []
# Попробовать создать nested run
class MLflowSource(Source):
    # todo: make it better
    """This is an MLflow Source"""

    platform = "mlflow"

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
        registered_models = self._get_mlflow_registered_models()
        for registered_model in registered_models:
            yield self._get_ml_group_workunit(registered_model)
            model_versions = self._get_mlflow_model_versions(registered_model)
            for model_version in model_versions:
                run = self._get_mlflow_run(model_version)
                yield self._get_ml_model_workunit(
                    registered_model=registered_model,
                    model_version=model_version,
                    run=run,
                )

    def _get_mlflow_registered_models(self) -> List[RegisteredModel]:
        # todo: implement pagination
        # todo: where all RegisteredModel properties like aliases come from?
        registered_models = self.client.search_registered_models()
        return registered_models

    def _get_mlflow_model_versions(self, registered_model: RegisteredModel) -> List[ModelVersion]:
        # todo: implement pagination
        filter_string = f"name = '{registered_model.name}'"
        model_versions = self.client.search_model_versions(filter_string)
        return model_versions

    def _get_mlflow_run(self, model_version: ModelVersion) -> Union[Run, None]:
        if model_version.run_id:
            run = self.client.get_run(model_version.run_id)
            return run
        else:
            return None

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

    # todo: replace List with Iterable?
    def _get_ml_group_workunit(self, registered_model: RegisteredModel) -> WorkUnit:
        ml_model_group_urn = builder.make_ml_model_group_urn(
            platform=self.platform,
            group_name=registered_model.name,
            env=self.env,
        )
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

    def _get_ml_model_workunit(
            self,
            registered_model: RegisteredModel,
            model_version: ModelVersion,
            run: Union[Run, None]
    ) -> WorkUnit:
        # we use mlflow registered model as a datahub ml model group
        ml_model_group_urn = builder.make_ml_model_group_urn(
            platform=self.platform,
            group_name=registered_model.name,
            env=self.env,
        )
        # we use each mlflow registered model version as a datahub ml model
        ml_model_urn = builder.make_ml_model_urn(
            platform=self.platform,
            model_name=f"{registered_model.name}_{model_version.version}",
            env=self.env,
        )
        # if a model was registered without an associated run then hyperparams and metrics are not available
        if run:
            hyperparams = [MLHyperParamClass(name=k, value=str(v)) for k, v in run.data.params.items()]
            training_metrics = [MLMetricClass(name=k, value=str(v)) for k, v in run.data.metrics.items()]
        else:
            hyperparams = None
            training_metrics = None
        # externalUrl: Union[None, str] = None,
        ml_model_properties = MLModelPropertiesClass(
            customProperties=model_version.tags,
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

    @classmethod
    def create(cls, config_dict: dict, ctx: PipelineContext) -> Source:
        config = MLflowConfig.parse_obj(config_dict)
        return cls(ctx, config)
