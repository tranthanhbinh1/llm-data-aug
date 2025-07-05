import dagster as dg
from src.worker.assets.iterative_optimization_v2 import iterative_optimization_result
from src.worker.resource import LLMResource, SynthesizerResource

# Create the main definitions object
defs = dg.Definitions(
    assets=[
        iterative_optimization_result,
    ],
    resources={
        "llm": LLMResource(api_key=dg.EnvVar("GOOGLE_AI_API_KEY")),
        "synthesizer": SynthesizerResource(),
    },
)


def create_repository():
    """Create the Dagster repository for LLM data augmentation pipeline."""
    return defs
