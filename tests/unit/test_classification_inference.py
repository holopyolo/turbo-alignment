from types import SimpleNamespace
from typing import Any

import pytest
import torch

from turbo_alignment.constants import MULTI_LABEL_CLASSIFICATION
from turbo_alignment.dataset.classification.models import ClassificationDatasetRecord
from turbo_alignment.generators.classification import ClassificationGenerator


class DummyTokenizer:
    padding_side = 'right'

    def pad(
        self,
        features: list[dict[str, torch.Tensor]],
        padding: bool = True,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | None = None,
    ) -> dict[str, torch.Tensor]:
        max_input_length = max(len(feature['input_ids']) for feature in features)
        padded_input_ids = []
        for feature in features:
            input_ids = feature['input_ids'].tolist()
            padded_input_ids.append(input_ids + [0] * (max_input_length - len(input_ids)))

        return {'input_ids': torch.tensor(padded_input_ids)}


class DummyClassificationModel:
    def __init__(self, logits: torch.Tensor, problem_type: str | None = None) -> None:
        self.config = SimpleNamespace(problem_type=problem_type)
        self.device = torch.device('cpu')
        self._logits = logits

    def __call__(self, **batch: torch.Tensor) -> SimpleNamespace:
        batch_size = batch['input_ids'].shape[0]
        return SimpleNamespace(logits=self._logits[:batch_size])


@pytest.mark.parametrize(
    ('problem_type', 'logits', 'expected_predictions'),
    [
        (
            None,
            torch.tensor([[0.1, 2.0], [3.0, 0.1]]),
            [1, 0],
        ),
        (
            MULTI_LABEL_CLASSIFICATION,
            torch.tensor([[10.0, -10.0, 0.0], [-10.0, 10.0, 10.0]]),
            [[1, 0, 1], [0, 1, 1]],
        ),
    ],
)
def test_classification_inference_passes_through_categories_and_labels(
    problem_type: str | None,
    logits: torch.Tensor,
    expected_predictions: list[int] | list[list[int]],
) -> None:
    records = [
        {'id': '0', 'input_ids': torch.tensor([1, 2, 3])},
        {'id': '1', 'input_ids': torch.tensor([4, 5])},
    ]
    metadata: list[dict[str, Any]] = [
        {'categories': {'kind': ['primary', 1]}, 'labels': ['external', 13]},
        {'categories': ['secondary', {'score': 0.7}], 'labels': {'flag': True}},
    ]
    original_records = [
        ClassificationDatasetRecord.model_validate(
            {
                'id': record['id'],
                'messages': [{'role': 'user', 'content': f'input {record["id"]}'}],
                'label': index,
                **metadata[index],
            }
        )
        for index, record in enumerate(records)
    ]
    generator = ClassificationGenerator(
        tokenizer=DummyTokenizer(),
        model=DummyClassificationModel(logits=logits, problem_type=problem_type),
        batch=2,
    )

    outputs = generator._generate_from_batch(records, original_records, 'classification_test')

    assert [output.predicted_label for output in outputs] == expected_predictions
    for output, expected_metadata in zip(outputs, metadata):
        assert output.categories == expected_metadata['categories']
        assert output.labels == expected_metadata['labels']
