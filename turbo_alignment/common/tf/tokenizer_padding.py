from collections.abc import Iterator
from typing import Any

from transformers import PreTrainedTokenizerBase

from turbo_alignment.common.logging import get_project_logger

logger = get_project_logger()

_QWEN_FLASH_ATTENTION_MODEL_TYPES = {'qwen2', 'qwen3'}


def _iter_wrapped_modules(model: Any) -> Iterator[Any]:
    seen: set[int] = set()
    stack = [model]

    while stack:
        module = stack.pop()
        if module is None:
            continue

        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)

        yield module

        for attr_name in ('module', 'base_model', 'model'):
            wrapped = getattr(module, attr_name, None)
            if wrapped is not None and wrapped is not module:
                stack.append(wrapped)


def model_requires_left_padding_for_flash_attention(model: Any) -> bool:
    for module in _iter_wrapped_modules(model):
        config = getattr(module, 'config', None)
        if config is None:
            continue

        attn_implementation = getattr(config, '_attn_implementation', None)
        model_type = getattr(config, 'model_type', None)
        if attn_implementation == 'flash_attention_2' and model_type in _QWEN_FLASH_ATTENTION_MODEL_TYPES:
            return True

    return False


def ensure_left_padding_for_flash_attention(
    tokenizer: PreTrainedTokenizerBase,
    model: Any,
) -> bool:
    if not model_requires_left_padding_for_flash_attention(model):
        return False

    if tokenizer.padding_side != 'left':
        logger.warning(
            f'Changing tokenizer.padding_side from "{tokenizer.padding_side}" to "left" because '
            'Qwen/Qwen3 with Flash Attention 2 requires left-padded batches.'
        )
        tokenizer.padding_side = 'left'

    return True
