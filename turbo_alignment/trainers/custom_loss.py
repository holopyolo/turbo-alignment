from typing import Callable

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    TrainingArguments,
)

from turbo_alignment.common.numeric_debug import NumericDebugState
from turbo_alignment.trainers.multigpu import MultiGPUCherryPicksTrainer


class CustomLossTrainer(MultiGPUCherryPicksTrainer):
    def __init__(
        self,
        model: PreTrainedModel | nn.Module,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        custom_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        data_collator: DataCollator,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        callbacks: list[TrainerCallback] | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        **kwargs,
    ):
        self.custom_loss = custom_loss
        self.numeric_debug = NumericDebugState(owner=type(self).__name__)
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            callbacks=callbacks,
            **kwargs,
        )
        self.numeric_debug.log_model_summary(self.model, args=self.args)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,  # pylint: disable=unused-argument
    ):
        """
        Modified original version, without manual label smoothing
        """
        if 'labels' in inputs:
            labels = inputs.pop('labels')
        else:
            raise ValueError('No labels provided in the inputs')

        outputs = model(**inputs)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]

        loss = self.custom_loss(logits, labels)
        call_idx = self.numeric_debug.next_forward_call()
        self.numeric_debug.log_forward(
            call_idx=call_idx,
            global_step=getattr(self.state, 'global_step', None),
            inputs=inputs,
            labels=labels,
            logits=logits,
            loss=loss,
        )

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        self.numeric_debug.scan_gradients(model, global_step=getattr(self.state, 'global_step', None))
        return loss
