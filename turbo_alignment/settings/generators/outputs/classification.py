from turbo_alignment.dataset.chat.models import ChatMessage
from turbo_alignment.settings.generators.outputs.base import BaseInferenceOutput


class ClassificationInferenceOutput(BaseInferenceOutput):
    id: str
    messages: list[ChatMessage]
    true_ground: int | None = None
    predicted_label: int
    class_probabilities: list[float]
