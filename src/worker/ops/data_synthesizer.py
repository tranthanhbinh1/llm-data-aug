import dagster as dg

from src.worker.resource import LLMResource


@dg.op
def synthesize_data(context: dg.OpExecutionContext, llm_resource: LLMResource):
