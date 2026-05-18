import math
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset
from transformers import EvalPrediction
from transformers.trainer import speed_metrics

from turbo_alignment.settings.pipelines.train.classification import (
    ClassificationLossSettings,
)
from turbo_alignment.trainers.custom_loss import CustomLossTrainer


def _safe_metric(metric_fn, *args, **kwargs) -> float:
    try:
        return float(metric_fn(*args, **kwargs))
    except ValueError:
        return float('nan')


def _roc_auc(labels: np.ndarray, pred_scores: np.ndarray) -> float:
    num_labels = pred_scores.shape[1]
    if num_labels == 2:
        return _safe_metric(roc_auc_score, labels, pred_scores[:, 1])

    return _safe_metric(
        roc_auc_score,
        labels,
        pred_scores,
        average='macro',
        multi_class='ovo',
        labels=np.arange(num_labels),
    )


def compute_clf_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    pred_scores, labels = eval_pred
    if isinstance(pred_scores, tuple):
        pred_scores = pred_scores[0]

    pred_scores = np.asarray(pred_scores)
    labels = np.asarray(labels)
    predictions = np.argmax(pred_scores, axis=1)

    num_labels = pred_scores.shape[1]
    average = 'binary' if num_labels == 2 else 'macro'

    accuracy = accuracy_score(labels, predictions)
    f_score = f1_score(labels, predictions, average=average, zero_division=0)
    precision = precision_score(labels, predictions, average=average, zero_division=0)
    recall = recall_score(labels, predictions, average=average, zero_division=0)
    roc_auc = _roc_auc(labels, pred_scores)
    specificity = (
        recall_score(labels, predictions, pos_label=0, zero_division=0) if num_labels == 2 else float('nan')
    )
    metrics = {
        'accuracy': accuracy,
        'specificity': specificity,
        'f1-score': f_score,
        'recall': recall,
        'precision': precision,
        'roc_auc': roc_auc,
    }
    return metrics


def classification_loss(
    logits: torch.Tensor, labels: torch.LongTensor, loss_settings: ClassificationLossSettings
) -> torch.Tensor:
    if loss_settings.alpha is None:
        alpha = torch.ones((logits.size(-1),), device=logits.device, dtype=logits.dtype)
    else:
        alpha = torch.tensor(loss_settings.alpha, device=logits.device, dtype=logits.dtype)

    ce_loss = F.cross_entropy(logits, labels, weight=alpha, reduction='none')

    p_t = torch.exp(-ce_loss)  # pylint: disable=invalid-unary-operand-type

    focal_loss = ((1 - p_t) ** loss_settings.gamma) * ce_loss

    return focal_loss.mean()


def auto_class_weights(dataset: Dataset) -> list[float]:
    labels = [dataset[i]['labels'] for i in range(len(dataset))]  # type: ignore[arg-type]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=np.array(labels))
    return class_weights.tolist()


class ClassificationTrainer(CustomLossTrainer):
    def __init__(
        self,
        loss_settings: ClassificationLossSettings,
        eval_dataset_slices: dict[str, Dataset] | None = None,
        **kwargs,
    ):
        self.eval_dataset_slices = eval_dataset_slices or {}
        super().__init__(
            custom_loss=partial(classification_loss, loss_settings=loss_settings),
            compute_metrics=compute_clf_metrics,
            **kwargs,
        )

    def evaluate(
        self,
        eval_dataset: Dataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = 'eval',
    ) -> dict[str, float]:
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if eval_dataset is None and metric_key_prefix == 'eval':
            for dataset_name, dataset in self.eval_dataset_slices.items():
                metrics.update(
                    self._evaluate_dataset_slice(
                        dataset=dataset,
                        ignore_keys=ignore_keys,
                        metric_key_prefix=f'{metric_key_prefix}_{dataset_name}',
                    )
                )

        return metrics

    def _evaluate_dataset_slice(
        self,
        dataset: Dataset,
        ignore_keys: list[str] | None,
        metric_key_prefix: str,
    ) -> dict[str, float]:
        eval_dataloader = self.get_eval_dataloader(dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description=f'Evaluation {metric_key_prefix}',
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / max(total_batch_size, 1)),
            )
        )

        self.log(output.metrics)

        return output.metrics
