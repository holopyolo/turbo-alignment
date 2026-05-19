from types import SimpleNamespace

from turbo_alignment.common.tf.tokenizer_padding import ensure_left_padding_for_flash_attention


def _model(model_type: str, attn_implementation: str):
    return SimpleNamespace(
        config=SimpleNamespace(
            model_type=model_type,
            _attn_implementation=attn_implementation,
        )
    )


def test_qwen_flash_attention_sets_left_padding() -> None:
    tokenizer = SimpleNamespace(padding_side='right')

    assert ensure_left_padding_for_flash_attention(tokenizer, _model('qwen3', 'flash_attention_2'))

    assert tokenizer.padding_side == 'left'


def test_wrapped_qwen_flash_attention_sets_left_padding() -> None:
    tokenizer = SimpleNamespace(padding_side='right')
    wrapped_model = SimpleNamespace(
        module=SimpleNamespace(
            base_model=SimpleNamespace(
                model=_model('qwen2', 'flash_attention_2'),
            )
        )
    )

    assert ensure_left_padding_for_flash_attention(tokenizer, wrapped_model)

    assert tokenizer.padding_side == 'left'


def test_non_qwen_flash_attention_keeps_padding_side() -> None:
    tokenizer = SimpleNamespace(padding_side='right')

    assert not ensure_left_padding_for_flash_attention(tokenizer, _model('bert', 'flash_attention_2'))

    assert tokenizer.padding_side == 'right'
