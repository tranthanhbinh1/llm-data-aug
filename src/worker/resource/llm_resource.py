import dagster as dg
from google import genai
import instructor


class LLMResource(dg.ConfigurableResource):
    api_key: str

    def get_instructor_instance(self):
        return instructor.from_genai(genai.Client(api_key=self.api_key))
