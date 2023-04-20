from typing import Iterable

from datahub.configuration.common import ConfigModel
from datahub.ingestion.api.common import PipelineContext, WorkUnit
from datahub.ingestion.api.source import Source, SourceReport


class MLflowConfig(ConfigModel):
    env: str = "PROD"


class MLflowSource(Source):
    report: SourceReport = SourceReport()

    def get_report(self) -> SourceReport:
        return self.report

    def get_workunits(self) -> Iterable[WorkUnit]:
        return []

    @classmethod
    def create(cls, config_dict: dict, ctx: PipelineContext) -> "Source":
        return cls(ctx)
