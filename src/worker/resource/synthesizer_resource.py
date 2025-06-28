from dagster import ConfigurableResource
from src.synthesizer.aug_gpt import AugGpt
from src.utils import get_instructor_instance


class SynthesizerResource(ConfigurableResource):
    def get_synthesizer_instance(self):
        return AugGpt(instructor=get_instructor_instance())
