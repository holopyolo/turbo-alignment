from typing import Optional
import json
from transformers import PreTrainedTokenizerBase

from turbo_alignment.metrics.metric import Metric
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults, MetricType


@Metric.register(MetricType.CUSTOM_LABEL)
class CustomMetric(Metric):
    def compute(self, **kwargs) -> list[MetricResults]:
        tokenizer: Optional[PreTrainedTokenizerBase] = kwargs.get('tokenizer', None)  # type: ignore[assignment]
        references: Optional[list[list[str]]] = kwargs.get('references', None)  # type: ignore[assignment]
        predictions: Optional[list[list[str]]] = kwargs.get('predictions', None)  # type: ignore[assignment]
        dataset_name: str = kwargs.get('dataset_name', '')

        if references is None:
            raise ValueError('references should not be None')

        if predictions is None:
            raise ValueError('predictions should not be None')

        if tokenizer is None:
            raise ValueError('tokenizer should not be None')

        return [
            MetricResults(
                element_wise_scores=[
                    ElementWiseScores(
                        label=dataset_name + '@@' + 'accuracy',
                        values=[
                            self._calculate_accuracy(
                                reference,
                                self._remove_suffix(prediction, tokenizer),
                            )
                            for reference_list, prediction_list in zip(references, predictions)
                            for reference, prediction in zip(reference_list, prediction_list)
                        ],
                    )
                ],
                need_average=need_average,
            )
            for need_average in self._settings.need_average
        ]
    @staticmethod
    def _calculate_accuracy(reference: str, prediction: str) -> float:
        reference_json = CustomMetric._load_json(reference)
        prediction_json = CustomMetric._load_json(prediction)
        total_true_match = 0
        if reference_json and prediction_json:
            # Check if both have the same criteria keys
            maps_value = {"true_debug": 'true'}
            reference_criteria = {k: str(v).lower() for k, v in reference_json.items() if k.startswith("criteria_")}
            prediction_criteria = {k: str(v).lower() for k, v in prediction_json.items() if k.startswith("criteria_")}
            
            reference_criteria = {k: maps_value.get(v, v) for k, v in reference_criteria.items()}
            prediction_criteria = {k: maps_value.get(v, v) for k, v in prediction_criteria.items()}
            
            # Check each criterion matches
            for key, ref_value in reference_criteria.items():
                if key in prediction_criteria and prediction_criteria[key] == ref_value:
                    total_true_match += 1
            return total_true_match / max(len(reference_criteria), 1)
        return 0.0



    @staticmethod
    def _load_json(s: str) -> dict:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return {}


    @staticmethod
    def _remove_suffix(prediction: str, tokenizer: PreTrainedTokenizerBase) -> str:
        return prediction.removesuffix(tokenizer.pad_token).removesuffix(tokenizer.eos_token)
