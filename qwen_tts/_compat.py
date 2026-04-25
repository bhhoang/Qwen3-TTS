# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Compatibility shims for older versions of transformers (< 4.52).

This module provides fallback implementations of APIs introduced in newer
versions of the transformers library so that the package remains installable
and functional on Python 3.8+ with transformers >= 4.40.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# No-op decorator helpers
# ---------------------------------------------------------------------------

def _noop_decorator(fn: Callable) -> Callable:
    """Return *fn* unchanged (identity decorator)."""
    return fn


def _noop_decorator_with_args(*args, **kwargs) -> Callable:
    """
    Decorator factory that ignores all arguments and returns an identity
    decorator.  Handles both ``@deco`` and ``@deco(...)`` call styles.
    """
    if len(args) == 1 and callable(args[0]) and not kwargs:
        # Used as @deco (no parentheses)
        return args[0]
    # Used as @deco(...) – return the identity decorator
    return _noop_decorator


# ---------------------------------------------------------------------------
# auto_docstring
# ---------------------------------------------------------------------------
try:
    from transformers.utils import auto_docstring  # noqa: F401
except ImportError:
    auto_docstring = _noop_decorator_with_args  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# can_return_tuple
# ---------------------------------------------------------------------------
try:
    from transformers.utils import can_return_tuple  # noqa: F401
except ImportError:
    can_return_tuple = _noop_decorator  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# dynamic_rope_update
# ---------------------------------------------------------------------------
try:
    from transformers.modeling_rope_utils import dynamic_rope_update  # noqa: F401
except ImportError:
    dynamic_rope_update = _noop_decorator  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# use_kernel_forward_from_hub
# ---------------------------------------------------------------------------
try:
    from transformers.integrations import use_kernel_forward_from_hub  # noqa: F401
except ImportError:
    def use_kernel_forward_from_hub(name: str) -> Callable:  # type: ignore[misc]
        """No-op – kernel hub loading is not available."""
        return _noop_decorator

# ---------------------------------------------------------------------------
# deprecate_kwarg
# ---------------------------------------------------------------------------
try:
    from transformers.utils.deprecation import deprecate_kwarg  # noqa: F401
except ImportError:
    def deprecate_kwarg(*args, **kwargs) -> Callable:  # type: ignore[misc]
        """No-op – deprecation decorator is not available."""
        return _noop_decorator

# ---------------------------------------------------------------------------
# check_model_inputs
# ---------------------------------------------------------------------------
try:
    from transformers.utils.generic import check_model_inputs  # noqa: F401
except ImportError:
    def check_model_inputs(*args, **kwargs) -> Callable:  # type: ignore[misc]
        """No-op – model-input validation decorator is not available."""
        return _noop_decorator

# ---------------------------------------------------------------------------
# layer_type_validation
# ---------------------------------------------------------------------------
try:
    from transformers.configuration_utils import layer_type_validation  # noqa: F401
except ImportError:
    def layer_type_validation(layer_types) -> None:  # type: ignore[misc]
        """No-op – layer-type validation is not available."""
        pass

# ---------------------------------------------------------------------------
# rope_config_validation
# ---------------------------------------------------------------------------
try:
    from transformers.modeling_rope_utils import rope_config_validation  # noqa: F401
except ImportError:
    def rope_config_validation(config) -> None:  # type: ignore[misc]
        """No-op – RoPE config validation is not available."""
        pass

# ---------------------------------------------------------------------------
# GradientCheckpointingLayer
# ---------------------------------------------------------------------------
try:
    from transformers.modeling_layers import GradientCheckpointingLayer  # noqa: F401
except ImportError:
    class GradientCheckpointingLayer(nn.Module):  # type: ignore[no-redef]
        """Fallback base class – gradient checkpointing is disabled."""
        pass

# ---------------------------------------------------------------------------
# FlashAttentionKwargs
# ---------------------------------------------------------------------------
try:
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs  # noqa: F401
except ImportError:
    try:
        from typing import TypedDict

        class FlashAttentionKwargs(TypedDict, total=False):  # type: ignore[no-redef]
            """Stub TypedDict for flash-attention keyword arguments."""
            pass
    except ImportError:
        FlashAttentionKwargs = dict  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Unpack
# ---------------------------------------------------------------------------
try:
    from transformers.processing_utils import Unpack  # noqa: F401
except ImportError:
    try:
        from typing import Unpack  # type: ignore[attr-defined]  # Python 3.11+
    except ImportError:
        try:
            from typing_extensions import Unpack  # noqa: F401
        except ImportError:
            # Annotations with Unpack are guarded by `from __future__ import
            # annotations`, so the symbol is never evaluated at runtime.
            Unpack = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# ALL_ATTENTION_FUNCTIONS
# ---------------------------------------------------------------------------

def _eager_attention_compat(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    is_causal: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pure-PyTorch eager attention (fallback for ALL_ATTENTION_FUNCTIONS)."""
    head_dim = query.shape[-1]
    scale = scaling if scaling is not None else (head_dim ** -0.5)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale

    if attention_mask is not None:
        # Additive mask of shape (B, 1, S, S) or (B, 1, 1, S)
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask
    elif is_causal:
        seq_len = query.shape[2]
        kv_len = key.shape[2]
        causal_mask = torch.triu(
            torch.full((seq_len, kv_len), float("-inf"), device=query.device, dtype=query.dtype),
            diagonal=1,
        )
        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout > 0.0 and module.training:
        attn_weights = F.dropout(attn_weights, p=dropout)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _sdpa_attention_compat(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    is_causal: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """SDPA-backed attention (fallback for ALL_ATTENTION_FUNCTIONS)."""
    head_dim = query.shape[-1]
    scale = scaling if scaling is not None else (head_dim ** -0.5)

    # Trim mask to actual kv length
    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    use_is_causal = is_causal and attention_mask is None

    try:
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout if module.training else 0.0,
            is_causal=use_is_causal,
            scale=scale,
        )
    except TypeError:
        # PyTorch < 2.1 doesn't accept the `scale` keyword argument
        query = query * scale
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout if module.training else 0.0,
            is_causal=use_is_causal,
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


class _AttentionInterface(dict):
    """Minimal dict-like registry compatible with transformers' AttentionInterface."""

    def valid_keys(self):
        return list(self.keys())


_FALLBACK_ATTENTION_FUNCTIONS: _AttentionInterface = _AttentionInterface(
    {
        "eager": _eager_attention_compat,
        "sdpa": _sdpa_attention_compat,
    }
)

try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: F401
except ImportError:
    ALL_ATTENTION_FUNCTIONS = _FALLBACK_ATTENTION_FUNCTIONS  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# create_causal_mask / create_sliding_window_causal_mask
# ---------------------------------------------------------------------------

def _create_causal_mask_pure(
    config,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values=None,
    sliding_window: Optional[int] = None,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Pure-PyTorch causal mask creation – used when neither the new
    ``transformers.masking_utils`` nor the legacy
    ``transformers.modeling_attn_mask_utils`` helper is available.
    """
    bsz, seq_len, _ = input_embeds.shape
    dtype = input_embeds.dtype
    device = input_embeds.device
    min_dtype = torch.finfo(dtype).min

    past_len = int(cache_position[0].item()) if cache_position is not None else 0
    total_len = past_len + seq_len

    # Build causal mask of shape (seq_len, total_len)
    causal_mask = torch.full((seq_len, total_len), fill_value=min_dtype, dtype=dtype, device=device)
    if seq_len > 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
        if cache_position is not None:
            # For each query position i, attend to positions <= cache_position[i]
            query_positions = cache_position.reshape(-1, 1)  # (seq_len, 1)
            kv_positions = torch.arange(total_len, device=device).unsqueeze(0)  # (1, total_len)
            future_mask = kv_positions > query_positions  # (seq_len, total_len)
            causal_mask = torch.where(future_mask, torch.full_like(causal_mask, min_dtype), torch.zeros_like(causal_mask))

    # Apply sliding window masking
    if sliding_window is not None:
        if cache_position is not None:
            query_positions = cache_position.reshape(-1, 1)
            kv_positions = torch.arange(total_len, device=device).unsqueeze(0)
            outside_window = kv_positions < (query_positions - sliding_window + 1)
        else:
            q_idx = torch.arange(seq_len, device=device).reshape(-1, 1)
            kv_idx = torch.arange(total_len, device=device).unsqueeze(0)
            outside_window = kv_idx < (q_idx + past_len - sliding_window + 1)
        causal_mask = causal_mask.masked_fill(outside_window, min_dtype)

    # Expand to (batch, 1, seq_len, total_len)
    causal_mask = causal_mask[None, None, :, :].expand(bsz, 1, -1, -1)

    # Incorporate the 2-D padding attention_mask
    if attention_mask is not None and attention_mask.dim() == 2:
        causal_mask = causal_mask.clone()
        mask_length = attention_mask.shape[-1]
        # Positions where attention_mask == 0 are padding – mask them out
        padding_mask = (
            causal_mask[..., :mask_length].eq(0.0)
            & attention_mask[:, None, None, :].eq(0.0)
        )
        causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(
            padding_mask, min_dtype
        )

    return causal_mask


def _create_causal_mask_legacy(
    config,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values=None,
    sliding_window: Optional[int] = None,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Create a causal mask using the legacy transformers (< 4.53) helper
    ``_prepare_4d_causal_attention_mask``.
    """
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask  # type: ignore[import]

    bsz, seq_len, _ = input_embeds.shape
    past_len = int(cache_position[0].item()) if cache_position is not None else 0
    return _prepare_4d_causal_attention_mask(
        attention_mask,
        (bsz, seq_len),
        input_embeds,
        past_len,
        sliding_window=sliding_window,
    )


try:
    from transformers.masking_utils import (  # noqa: F401
        create_causal_mask,
        create_sliding_window_causal_mask,
    )
except ImportError:
    try:
        # transformers 4.39 – 4.52: legacy helper is available
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask  # noqa: F401

        def create_causal_mask(  # type: ignore[misc]
            config,
            input_embeds: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            cache_position: torch.Tensor,
            past_key_values=None,
            **kwargs,
        ) -> Optional[torch.Tensor]:
            return _create_causal_mask_legacy(
                config, input_embeds, attention_mask, cache_position, past_key_values
            )

        def create_sliding_window_causal_mask(  # type: ignore[misc]
            config,
            input_embeds: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            cache_position: torch.Tensor,
            past_key_values=None,
            **kwargs,
        ) -> Optional[torch.Tensor]:
            sliding_window = getattr(config, "sliding_window", None)
            return _create_causal_mask_legacy(
                config, input_embeds, attention_mask, cache_position, past_key_values,
                sliding_window=sliding_window,
            )

    except ImportError:
        # Very old transformers – pure-PyTorch fallback
        def create_causal_mask(  # type: ignore[misc]
            config,
            input_embeds: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            cache_position: torch.Tensor,
            past_key_values=None,
            **kwargs,
        ) -> Optional[torch.Tensor]:
            return _create_causal_mask_pure(
                config, input_embeds, attention_mask, cache_position, past_key_values
            )

        def create_sliding_window_causal_mask(  # type: ignore[misc]
            config,
            input_embeds: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            cache_position: torch.Tensor,
            past_key_values=None,
            **kwargs,
        ) -> Optional[torch.Tensor]:
            sliding_window = getattr(config, "sliding_window", None)
            return _create_causal_mask_pure(
                config, input_embeds, attention_mask, cache_position, past_key_values,
                sliding_window=sliding_window,
            )
