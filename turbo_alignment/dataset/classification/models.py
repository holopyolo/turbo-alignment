from typing import Any

from turbo_alignment.dataset.base.models import DatasetRecord
from turbo_alignment.dataset.chat.models import ChatMessage


class ClassificationDatasetRecord(DatasetRecord):
    messages: list[ChatMessage]
    label: int | list[int] | None
    categories: Any | None = None
    labels: Any | None = None
