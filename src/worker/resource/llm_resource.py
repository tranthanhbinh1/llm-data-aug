import dagster as dg
from src.utils import get_instructor_instance


class LLMResource(dg.ConfigurableResource):
    api_key: str

    def get_instructor_instance(self):
        return get_instructor_instance()
