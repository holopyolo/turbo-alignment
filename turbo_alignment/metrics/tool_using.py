import torch
from transformers import PreTrainedTokenizerBase

from turbo_alignment.metrics.metric import Metric
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults, MetricType


@Metric.register(MetricType.LENGTH)
class ToolMetric(Metric):
    def compute(self, **kwargs) -> list[MetricResults]:
        tokenizer: PreTrainedTokenizerBase = kwargs.get('tokenizer', None)
        predictions: str = kwargs.get('predictions', None)
        references: str = kwargs.get('references', '')
        dataset_name: str = kwargs.get('dataset_name', '')
        return [
            MetricResults(
                element_wise_scores=[
                    ElementWiseScores(
                        label=dataset_name + '@@' + 'length',
                        values=[
                           references == predictions
                        ],
                    )
                ],
                need_average=need_average,
            )
            for need_average in self._settings.need_average
        ]
