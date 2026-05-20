from __future__ import annotations

import os
import time
from collections import Counter
from typing import Any

import torch
from torch import nn


TRUE_VALUES = {'1', 'true', 'yes', 'y', 'on'}
_PARAM_SCAN_COUNTS: Counter[str] = Counter()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in TRUE_VALUES


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def numeric_debug_enabled() -> bool:
    return _env_bool('TURBO_ALIGNMENT_NUMERIC_DEBUG') or _env_bool('TA_NUMERIC_DEBUG')


def _rank() -> str:
    return os.getenv('RANK') or os.getenv('LOCAL_RANK') or '?'


def _local_rank() -> str:
    return os.getenv('LOCAL_RANK') or '?'


def _rank0_only() -> bool:
    return _env_bool('TURBO_ALIGNMENT_NUMERIC_DEBUG_RANK0_ONLY') or _env_bool('TA_NUMERIC_DEBUG_RANK0_ONLY')


def _should_log(force: bool = False) -> bool:
    if force:
        return True
    if not _rank0_only():
        return True
    return _rank() in {'0', '?'}


def debug_log(message: str, *, force: bool = False) -> None:
    if not numeric_debug_enabled() or not _should_log(force=force):
        return

    timestamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    print(
        f'{timestamp} [TA_NUMERIC_DEBUG rank={_rank()} local_rank={_local_rank()}] {message}',
        flush=True,
    )


def _shape(tensor: torch.Tensor) -> str:
    return '[' + ','.join(str(dim) for dim in tensor.shape) + ']'


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def _format_float(value: float) -> str:
    return f'{value:.6g}'


def _tensor_stats(name: str, tensor: Any) -> str:
    if not torch.is_tensor(tensor):
        return f'{name}=<{type(tensor).__name__}>'

    detached = tensor.detach()
    base = f'{name}(shape={_shape(detached)} dtype={detached.dtype} device={detached.device}'
    if detached.numel() == 0:
        return f'{base} empty=True)'

    with torch.no_grad():
        if detached.is_floating_point() or detached.is_complex():
            finite_mask = torch.isfinite(detached)
            finite_count = int(finite_mask.sum().cpu().item())
            total = detached.numel()
            nan_count = int(torch.isnan(detached).sum().cpu().item())
            inf_count = int(torch.isinf(detached).sum().cpu().item())
            parts = [
                base,
                f'finite={finite_count}/{total}',
                f'nan={nan_count}',
                f'inf={inf_count}',
            ]
            if finite_count > 0:
                values = detached[finite_mask].float()
                parts.extend(
                    [
                        f'min={_format_float(_to_float(values.min()))}',
                        f'max={_format_float(_to_float(values.max()))}',
                        f'absmax={_format_float(_to_float(values.abs().max()))}',
                        f'mean={_format_float(_to_float(values.mean()))}',
                    ]
                )
                if values.numel() > 1:
                    parts.append(f'std={_format_float(_to_float(values.std(unbiased=False)))}')
            return ' '.join(parts) + ')'

        parts = [base]
        if detached.numel() <= 16:
            parts.append(f'values={detached.cpu().tolist()}')
        else:
            parts.extend(
                [
                    f'min={_to_float(detached.min())}',
                    f'max={_to_float(detached.max())}',
                ]
            )
            if detached.dtype == torch.bool or detached.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                parts.append(f'sum={_to_float(detached.sum())}')
        return ' '.join(parts) + ')'


def _has_non_finite(tensor: torch.Tensor) -> bool:
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return False
    with torch.no_grad():
        return not bool(torch.isfinite(tensor.detach()).all().cpu().item())


def log_tensors(event: str, tensors: dict[str, Any], *, force: bool = False) -> None:
    if not numeric_debug_enabled():
        return

    is_bad = any(torch.is_tensor(tensor) and _has_non_finite(tensor) for tensor in tensors.values())
    should_log = force or is_bad or _env_bool('TURBO_ALIGNMENT_NUMERIC_DEBUG_LOSS_COMPONENTS')
    if not should_log:
        return

    debug_log(
        event + ' bad=' + str(is_bad) + ' ' + ' '.join(_tensor_stats(name, tensor) for name, tensor in tensors.items()),
        force=is_bad or force,
    )


def _optimizer_lrs(optimizer: Any) -> list[float | str]:
    param_groups = getattr(optimizer, 'param_groups', None)
    if param_groups is None and hasattr(optimizer, 'optimizer'):
        param_groups = getattr(optimizer.optimizer, 'param_groups', None)
    if param_groups is None:
        return []

    lrs = []
    for group in param_groups[:8]:
        lr = group.get('lr', '<missing>')
        if torch.is_tensor(lr):
            lr = _to_float(lr)
        lrs.append(lr)
    return lrs


def scan_parameters(model: nn.Module, *, stage: str, global_step: int | None) -> None:
    if not numeric_debug_enabled() or not _env_bool('TURBO_ALIGNMENT_NUMERIC_DEBUG_PARAMS', True):
        return

    _PARAM_SCAN_COUNTS[stage] += 1
    scan_idx = _PARAM_SCAN_COUNTS[stage]
    max_scans = max(0, _env_int('TURBO_ALIGNMENT_NUMERIC_DEBUG_MAX_PARAM_SCANS', 8))
    every = max(1, _env_int('TURBO_ALIGNMENT_NUMERIC_DEBUG_PARAM_EVERY', 1))
    if scan_idx > max_scans or scan_idx % every != 0:
        return

    checked_params = 0
    checked_elems = 0
    bad_param_parts: list[str] = []
    max_bad_params = max(1, _env_int('TURBO_ALIGNMENT_NUMERIC_DEBUG_MAX_BAD_PARAMS', 12))
    max_good_examples = max(0, _env_int('TURBO_ALIGNMENT_NUMERIC_DEBUG_MAX_GOOD_PARAM_EXAMPLES', 3))
    good_examples: list[str] = []

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            checked_params += 1
            checked_elems += parameter.numel()
            values = parameter.detach()
            if not (values.is_floating_point() or values.is_complex()):
                continue

            finite_mask = torch.isfinite(values)
            is_finite = bool(finite_mask.all().cpu().item())
            if is_finite:
                if len(good_examples) < max_good_examples:
                    good_examples.append(f'{name}:dtype={values.dtype} shape={_shape(values)}')
                continue

            finite_count = int(finite_mask.sum().cpu().item())
            nan_count = int(torch.isnan(values).sum().cpu().item())
            inf_count = int(torch.isinf(values).sum().cpu().item())
            finite_suffix = ''
            if finite_count > 0:
                finite_values = values[finite_mask].float()
                finite_suffix = f' finite_absmax={_format_float(_to_float(finite_values.abs().max()))}'

            bad_param_parts.append(
                f'{name}:shape={_shape(values)} dtype={values.dtype} '
                f'finite={finite_count}/{values.numel()} nan={nan_count} inf={inf_count}{finite_suffix}'
            )
            if len(bad_param_parts) >= max_bad_params:
                break

    has_bad_params = len(bad_param_parts) > 0
    debug_log(
        'param_scan '
        f'stage={stage} scan={scan_idx} global_step={global_step} bad={has_bad_params} '
        f'checked_params={checked_params} checked_elems={checked_elems} '
        f'good_examples={good_examples} bad_params={bad_param_parts}',
        force=has_bad_params,
    )


class NumericDebugState:
    def __init__(self, owner: str):
        self.owner = owner
        self.forward_calls = 0
        self.grad_scans = 0
        self.model_summary_logged = False
        self.grad_hooks_registered = False
        self.grad_hook_logs = 0
        self.grad_hook_handles: list[Any] = []
        self.get_global_step = None
        self.every = max(1, _env_int('TURBO_ALIGNMENT_NUMERIC_DEBUG_EVERY', 1))
        self.max_forward_logs = max(0, _env_int('TURBO_ALIGNMENT_NUMERIC_DEBUG_MAX_FORWARD_LOGS', 20))
        self.grad_scan_enabled = _env_bool('TURBO_ALIGNMENT_NUMERIC_DEBUG_GRADS', True)
        self.grad_scan_every = max(1, _env_int('TURBO_ALIGNMENT_NUMERIC_DEBUG_GRAD_EVERY', 1))
        self.max_grad_scans = max(0, _env_int('TURBO_ALIGNMENT_NUMERIC_DEBUG_MAX_GRAD_SCANS', 20))
        self.max_bad_grad_params = max(1, _env_int('TURBO_ALIGNMENT_NUMERIC_DEBUG_MAX_BAD_GRAD_PARAMS', 8))
        self.max_grad_hook_logs = max(0, _env_int('TURBO_ALIGNMENT_NUMERIC_DEBUG_MAX_GRAD_HOOK_LOGS', 24))

    @property
    def enabled(self) -> bool:
        return numeric_debug_enabled()

    def log_model_summary(self, model: nn.Module, args: Any | None = None) -> None:
        if not self.enabled or self.model_summary_logged:
            return

        self.model_summary_logged = True
        total_params = 0
        trainable_params = 0
        dtype_counts: Counter[str] = Counter()
        trainable_dtype_counts: Counter[str] = Counter()
        trainable_examples: list[str] = []

        for name, parameter in model.named_parameters():
            numel = parameter.numel()
            dtype = str(parameter.dtype)
            total_params += numel
            dtype_counts[dtype] += numel
            if parameter.requires_grad:
                trainable_params += numel
                trainable_dtype_counts[dtype] += numel
                if len(trainable_examples) < 12:
                    trainable_examples.append(name)

        config = getattr(model, 'config', None)
        config_bits = []
        for field_name in ('model_type', 'torch_dtype', 'num_labels', 'problem_type', '_attn_implementation'):
            if config is not None and hasattr(config, field_name):
                config_bits.append(f'{field_name}={getattr(config, field_name)}')

        arg_bits = []
        for field_name in ('bf16', 'fp16', 'tf32', 'gradient_checkpointing', 'max_grad_norm', 'deepspeed'):
            if args is not None and hasattr(args, field_name):
                arg_bits.append(f'{field_name}={getattr(args, field_name)}')

        debug_log(
            'model_summary '
            f'owner={self.owner} model={type(model).__name__} '
            f'total_params={total_params} trainable_params={trainable_params} '
            f'dtypes={dict(dtype_counts)} trainable_dtypes={dict(trainable_dtype_counts)} '
            f'trainable_examples={trainable_examples} '
            f'config={";".join(config_bits)} args={";".join(arg_bits)}',
            force=True,
        )

    def register_gradient_hooks(self, model: nn.Module, get_global_step: Any | None = None) -> None:
        if (
            not self.enabled
            or self.grad_hooks_registered
            or not _env_bool('TURBO_ALIGNMENT_NUMERIC_DEBUG_GRAD_HOOKS', True)
            or self.max_grad_hook_logs == 0
        ):
            return

        self.grad_hooks_registered = True
        self.get_global_step = get_global_step
        registered = 0

        def make_hook(name: str):
            def hook(grad: torch.Tensor) -> torch.Tensor:
                if self.grad_hook_logs >= self.max_grad_hook_logs:
                    return grad
                if not torch.is_tensor(grad) or not _has_non_finite(grad):
                    return grad

                self.grad_hook_logs += 1
                global_step = self.get_global_step() if self.get_global_step is not None else None
                debug_log(
                    'param_grad_hook '
                    f'owner={self.owner} log={self.grad_hook_logs} global_step={global_step} '
                    f'param={name} '
                    + _tensor_stats('grad', grad),
                )
                return grad

            return hook

        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.grad_hook_handles.append(parameter.register_hook(make_hook(name)))
                registered += 1

        debug_log(
            f'grad_hooks_registered owner={self.owner} count={registered} max_logs={self.max_grad_hook_logs}',
            force=True,
        )

    def next_forward_call(self) -> int:
        self.forward_calls += 1
        return self.forward_calls

    def log_forward(
        self,
        *,
        call_idx: int,
        global_step: int | None,
        inputs: dict[str, Any],
        labels: torch.Tensor,
        logits: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return

        is_bad = _has_non_finite(logits) or _has_non_finite(loss)
        should_log = is_bad or (
            call_idx <= self.max_forward_logs and (call_idx == 1 or call_idx % self.every == 0)
        )
        if not should_log:
            return

        input_parts = []
        for key in ('input_ids', 'attention_mask', 'position_ids'):
            if key in inputs:
                input_parts.append(_tensor_stats(key, inputs[key]))

        debug_log(
            'forward '
            f'owner={self.owner} call={call_idx} global_step={global_step} bad={is_bad} '
            + ' '.join(input_parts)
            + ' '
            + _tensor_stats('labels', labels)
            + ' '
            + _tensor_stats('logits', logits)
            + ' '
            + _tensor_stats('loss', loss),
            force=is_bad,
        )

    def scan_gradients(self, model: nn.Module, *, global_step: int | None) -> None:
        if not self.enabled or not self.grad_scan_enabled:
            return

        self.grad_scans += 1
        if self.grad_scans > self.max_grad_scans or self.grad_scans % self.grad_scan_every != 0:
            return

        checked_params = 0
        checked_elems = 0
        none_grads = 0
        bad_grad_parts: list[str] = []

        with torch.no_grad():
            for name, parameter in model.named_parameters():
                grad = parameter.grad
                if grad is None:
                    none_grads += 1
                    continue

                checked_params += 1
                checked_elems += grad.numel()
                values = grad
                if values.is_sparse:
                    values = values.coalesce().values()

                if not (values.is_floating_point() or values.is_complex()):
                    continue

                finite_mask = torch.isfinite(values)
                if bool(finite_mask.all().cpu().item()):
                    continue

                finite_count = int(finite_mask.sum().cpu().item())
                nan_count = int(torch.isnan(values).sum().cpu().item())
                inf_count = int(torch.isinf(values).sum().cpu().item())
                finite_suffix = ''
                if finite_count > 0:
                    finite_values = values[finite_mask].float()
                    finite_suffix = f' finite_absmax={_format_float(_to_float(finite_values.abs().max()))}'

                bad_grad_parts.append(
                    f'{name}:shape={_shape(values)} dtype={values.dtype} '
                    f'finite={finite_count}/{values.numel()} nan={nan_count} inf={inf_count}{finite_suffix}'
                )
                if len(bad_grad_parts) >= self.max_bad_grad_params:
                    break

        has_bad_grads = len(bad_grad_parts) > 0
        debug_log(
            'grad_scan '
            f'owner={self.owner} scan={self.grad_scans} global_step={global_step} '
            f'bad={has_bad_grads} checked_params={checked_params} checked_elems={checked_elems} '
            f'none_grads={none_grads} bad_params={bad_grad_parts}',
            force=has_bad_grads,
        )


def log_optimizer_step(
    *,
    stage: str,
    global_step: int | None,
    grad_norm: Any = None,
    max_grad_norm: float | None = None,
    learning_rate: Any = None,
    optimizer_step_was_skipped: bool | None = None,
    optimizer: Any = None,
) -> None:
    if not numeric_debug_enabled():
        return

    debug_log(
        'optimizer '
        f'stage={stage} global_step={global_step} grad_norm={grad_norm} '
        f'max_grad_norm={max_grad_norm} learning_rate={learning_rate} '
        f'optimizer_lrs={_optimizer_lrs(optimizer) if optimizer is not None else []} '
        f'optimizer_step_was_skipped={optimizer_step_was_skipped}',
        force=bool(optimizer_step_was_skipped),
    )
