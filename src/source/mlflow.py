from typing import Iterable, Optional

from datahub.configuration.common import ConfigModel
from datahub.ingestion.api.common import PipelineContext, WorkUnit
from datahub.ingestion.api.source import Source, SourceReport
from mlflow import MlflowClient
from pydantic.fields import Field
from datahub.ingestion.api.workunit import MetadataWorkUnit
from datahub.emitter.mcp import MetadataChangeProposalWrapper
from datahub.metadata.schema_classes import MLModelPropertiesClass
import datahub.emitter.mce_builder as builder


class MLflowConfig(ConfigModel):
    tracking_uri: Optional[str] = Field(
        default=None,
        # https://mlflow.org/docs/latest/python_api/mlflow.html?highlight=registry_uri#mlflow.set_tracking_uri
        description="Tracking server URI"
    )
    registry_uri: Optional[str] = Field(
        default=None,
        # https://mlflow.org/docs/latest/python_api/mlflow.html?highlight=registry_uri#mlflow.set_registry_uri
        description="Registry server URI"
    )


class MLflowSource(Source):
    # todo: make it better
    """This is an MLflow Source"""

    platform = "mlflow"
    stage_map = {
        'Staging': 'STG',
        'Production': 'PROD',
    }
    stages_to_use = [
        "Staging",
        "Production",
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

    def get_report(self) -> SourceReport:
        return self.report

    def get_workunits(self) -> Iterable[WorkUnit]:
        # todo: implement pagination
        for registered_model in self.client.search_registered_models():
            print(f"Processing model: {registered_model.name}")
            model_versions = self.client.get_latest_versions(
                name=registered_model.name,
                stages=self.stages_to_use,
            )
            for model_version in model_versions:
                print(model_version.current_stage)
                ml_model_urn = builder.make_ml_model_urn(
                    platform=self.platform,
                    model_name=registered_model.name,
                    env=self.stage_map[model_version.current_stage],
                )
                # customProperties: Optional[Dict[str, str]] = None,
                # externalUrl: Union[None, str] = None,
                # description: Union[None, str] = None,
                # date: Union[None, int] = None,
                # version: Union[None, "VersionTagClass"] = None,
                # type: Union[None, str] = None,
                # hyperParams: Union[None, List["MLHyperParamClass"]] = None,
                # trainingMetrics: Union[None, List["MLMetricClass"]] = None,
                # mlFeatures: Union[None, List[str]] = None,
                # tags: Optional[List[str]] = None,
                # deployments: Union[None, List[str]] = None,
                # groups: Union[None, List[str]] = None,
                ml_model_properties = MLModelPropertiesClass(
                    # todo: do smth?
                    customProperties=registered_model.tags,
                    description=registered_model.description,
                    date=registered_model.creation_timestamp,
                    # todo: do smth?
                    # mlflow tags are dicts, but datahub tags are lists. currently use only keys from mlflow tags
                    tags=list(registered_model.tags),
                )
                mcp = MetadataChangeProposalWrapper(
                    entityUrn=ml_model_urn,
                    aspect=ml_model_properties,
                )
                wu = MetadataWorkUnit(
                    # don't understand a purpose of this id
                    id=f"{registered_model.name}_{model_version.current_stage}",
                    mcp=mcp,
                )
                self.report.report_workunit(wu)
                yield wu

    @classmethod
    def create(cls, config_dict: dict, ctx: PipelineContext) -> Source:
        config = MLflowConfig.parse_obj(config_dict)
        return cls(ctx, config)
