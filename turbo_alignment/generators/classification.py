from typing import Any

import torch
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from turbo_alignment.common.tf.tokenizer_padding import ensure_left_padding_for_flash_attention
from turbo_alignment.constants import MULTI_LABEL_CLASSIFICATION
from turbo_alignment.dataset.classification.models import ClassificationDatasetRecord
from turbo_alignment.generators.base import BaseGenerator
from turbo_alignment.settings.generators.outputs.classification import (
    ClassificationInferenceOutput,
)


class ClassificationGenerator(BaseGenerator[ClassificationDatasetRecord, ClassificationInferenceOutput]):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)

        ensure_left_padding_for_flash_attention(tokenizer, self._model)
        self._collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    def _is_multi_label_classification(self) -> bool:
        models = [
            self._model,
            getattr(self._model, 'module', None),
            getattr(getattr(self._model, 'base_model', None), 'model', None),
            getattr(getattr(getattr(self._model, 'module', None), 'base_model', None), 'model', None),
        ]
        for model in models:
            config = getattr(model, 'config', None)
            if getattr(config, 'problem_type', None) == MULTI_LABEL_CLASSIFICATION:
                return True

        return False

    def _generate_from_batch(
        self, records: list[dict[str, Any]], original_records: list[ClassificationDatasetRecord], dataset_name: str
    ) -> list[ClassificationInferenceOutput]:
        inputs = [{'input_ids': record['input_ids']} for record in records]
        is_multi_label = self._is_multi_label_classification()

        with torch.no_grad():
            batch = {key: value.to(self.device) for key, value in self._collator(inputs).items()}
            output_logits = self._model(**batch).logits
            if is_multi_label:
                probabilities = torch.sigmoid(output_logits)
                classes = (probabilities >= 0.5).to(torch.int)
            else:
                probabilities = torch.softmax(output_logits, dim=1)
                classes = torch.argmax(output_logits, dim=1)

        return [
            ClassificationInferenceOutput(
                id=record.id,
                messages=record.messages,
                true_ground=record.label,
                predicted_label=cl.tolist() if is_multi_label else cl.item(),
                class_probabilities=probs.tolist(),
                dataset_name=dataset_name,
            )
            for record, cl, probs in zip(original_records, classes, probabilities)
        ]
