from typing import Any

from turbo_alignment.dataset.chat.models import ChatMessage
from turbo_alignment.settings.generators.outputs.base import BaseInferenceOutput


class ClassificationInferenceOutput(BaseInferenceOutput):
    id: str
    messages: list[ChatMessage]
    true_ground: int | list[int] | None = None
    predicted_label: int | list[int]
    class_probabilities: list[float]
    categories: Any | None = None
    labels: Any | None = None
