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
    # todo: ???
    env: str = "PROD"


class MLflowSource(Source):
    # todo: make it better
    """This is an MLflow Source"""

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
        # todo: implement pagination
        for rm in self.client.search_registered_models():
            print(dict(rm))
            ml_model_urn = builder.make_ml_model_urn(
                platform="MLflow",
                model_name=rm.name,
                env=self.env,
            )
            # customProperties: Optional[Dict[str, str]] = None,
            # externalUrl: Union[None, str] = None,
            # description: Union[None, str] = None,
            # date: Union[None, int] = None,
            # version: Union[None, "VersionTagClass"] = None,
            # type: Union[None, str] = None,
            # hyperParameters: Union[None, Dict[str, Union[str, int, float, float, bool]]] = None,
            # hyperParams: Union[None, List["MLHyperParamClass"]] = None,
            # trainingMetrics: Union[None, List["MLMetricClass"]] = None,
            # onlineMetrics: Union[None, List["MLMetricClass"]] = None,
            # mlFeatures: Union[None, List[str]] = None,
            # tags: Optional[List[str]] = None,
            # deployments: Union[None, List[str]] = None,
            # trainingJobs: Union[None, List[str]] = None,
            # downstreamJobs: Union[None, List[str]] = None,
            # groups: Union[None, List[str]] = None,
            ml_model_properties = MLModelPropertiesClass(
                # todo: do smth?
                customProperties=rm.tags,
                description=rm.description,
                date=rm.creation_timestamp,
                # todo: do smth?
                # mlflow tags are dicts, but datahub tags are lists. currently use only keys from mlflow tags
                tags=list(rm.tags),
            )
            mcp = MetadataChangeProposalWrapper(
                entityUrn=ml_model_urn,
                aspect=ml_model_properties,
            )
            wu = MetadataWorkUnit(
                # don't understand a purpose of this id
                id="id__" + rm.name,
                mcp=mcp,
            )
            self.report.report_workunit(wu)
            yield wu

    @classmethod
    def create(cls, config_dict: dict, ctx: PipelineContext) -> Source:
        config = MLflowConfig.parse_obj(config_dict)
        return cls(ctx, config)
