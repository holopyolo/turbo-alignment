import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from turbo_alignment.dataset.classification.models import ClassificationDatasetRecord
from turbo_alignment.settings.generators.outputs.classification import (
    ClassificationInferenceOutput,
)
from turbo_alignment.settings.pipelines.train.classification import (
    ClassificationLossSettings,
)
from turbo_alignment.trainers.classification import (
    MULTI_LABEL_CLASSIFICATION,
    auto_class_weights,
    classification_loss,
    compute_clf_metrics,
)

REMOVED_CLEARML_METRICS = {'accuracy', 'f1-score', 'roc_auc', 'specificity'}


class LabelsDataset(Dataset):
    def __init__(self, labels: list[list[int]]) -> None:
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return {'labels': self.labels[index]}


def test_classification_models_accept_multilabel() -> None:
    record = ClassificationDatasetRecord.model_validate(
        {
            'id': '0',
            'messages': [{'role': 'user', 'content': 'hi'}],
            'label': [1, 0, 1],
        }
    )
    output = ClassificationInferenceOutput.model_validate(
        {
            'id': '0',
            'messages': [{'role': 'user', 'content': 'hi'}],
            'true_ground': [1, 0, 1],
            'predicted_label': [1, 0, 0],
            'class_probabilities': [0.8, 0.2, 0.4],
            'dataset_name': 'classification_multilabel_test',
        }
    )

    assert record.label == [1, 0, 1]
    assert output.true_ground == [1, 0, 1]
    assert output.predicted_label == [1, 0, 0]


def test_multilabel_loss_matches_bce_with_logits() -> None:
    logits = torch.tensor([[1.0, -1.0, 0.5], [-0.5, 0.25, 1.5]])
    labels = torch.tensor([[1, 0, 1], [0, 1, 0]])
    loss_settings = ClassificationLossSettings(alpha=None, gamma=10.0)

    actual_loss = classification_loss(
        logits=logits,
        labels=labels,
        loss_settings=loss_settings,
        problem_type=MULTI_LABEL_CLASSIFICATION,
    )
    expected_loss = F.binary_cross_entropy_with_logits(logits, labels.float())

    assert torch.allclose(actual_loss, expected_loss)


def test_multilabel_loss_uses_alpha_as_pos_weight() -> None:
    logits = torch.tensor([[1.0, -1.0, 0.5], [-0.5, 0.25, 1.5]])
    labels = torch.tensor([[1, 0, 1], [0, 1, 0]])
    pos_weight = torch.tensor([0.5, 2.0, 1.5])
    loss_settings = ClassificationLossSettings(alpha=pos_weight.tolist(), gamma=10.0)

    actual_loss = classification_loss(
        logits=logits,
        labels=labels,
        loss_settings=loss_settings,
        problem_type=MULTI_LABEL_CLASSIFICATION,
    )
    expected_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)

    assert torch.allclose(actual_loss, expected_loss)


def test_multilabel_auto_class_weights_are_pos_weights() -> None:
    dataset = LabelsDataset(labels=[[1, 0, 1], [0, 1, 0], [1, 1, 0]])

    weights = auto_class_weights(dataset, problem_type=MULTI_LABEL_CLASSIFICATION)

    assert weights == pytest.approx([0.5, 0.5, 2.0])


def test_auto_class_weights_detects_multilabel_labels_without_problem_type() -> None:
    dataset = LabelsDataset(labels=[[1, 0, 1], [0, 1, 0], [1, 1, 0]])

    weights = auto_class_weights(dataset)

    assert weights == pytest.approx([0.5, 0.5, 2.0])


def test_multilabel_metrics_threshold_sigmoid_at_half() -> None:
    logits = np.array(
        [
            [2.0, -2.0, 1.0],
            [-1.0, 2.0, -3.0],
            [0.1, 0.2, -0.1],
            [-0.2, -0.3, 0.5],
        ]
    )
    labels = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ]
    )

    metrics = compute_clf_metrics((logits, labels), problem_type=MULTI_LABEL_CLASSIFICATION)

    assert metrics['recall'] == 1.0
    assert metrics['precision'] == 1.0
    assert metrics['average_precision'] == 1.0
    assert metrics['fpr'] == 0.0
    assert REMOVED_CLEARML_METRICS.isdisjoint(metrics)


def test_binary_classification_metrics_include_fpr() -> None:
    logits = np.array(
        [
            [0.0, 2.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [0.0, 2.0],
        ]
    )
    labels = np.array([0, 0, 0, 1])

    metrics = compute_clf_metrics((logits, labels))

    assert metrics['fpr'] == pytest.approx(2 / 3)
    assert metrics['average_precision'] == pytest.approx(1 / 3)
    assert REMOVED_CLEARML_METRICS.isdisjoint(metrics)


def test_multiclass_classification_metrics_include_macro_average_precision() -> None:
    logits = np.array(
        [
            [3.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 3.0],
            [2.0, 1.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.0, 1.0, 2.0],
        ]
    )
    labels = np.array([0, 1, 2, 0, 1, 2])

    metrics = compute_clf_metrics((logits, labels))

    assert metrics['average_precision'] == 1.0
    assert REMOVED_CLEARML_METRICS.isdisjoint(metrics)


def test_multilabel_metrics_include_macro_fpr() -> None:
    logits = np.array(
        [
            [2.0, 2.0, -2.0],
            [2.0, -2.0, 2.0],
            [-2.0, 2.0, 2.0],
            [-2.0, -2.0, -2.0],
        ]
    )
    labels = np.array(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 0],
        ]
    )

    metrics = compute_clf_metrics((logits, labels), problem_type=MULTI_LABEL_CLASSIFICATION)

    assert metrics['fpr'] == pytest.approx(1 / 3)
