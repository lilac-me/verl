# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Eagle3 draft model wrapper for verl FSDP training.

Loads an Eagle3 HuggingFace checkpoint and exposes it as a standard
``nn.Module`` compatible with FSDP / DDP sharding.

The Eagle3 forward signature expected from the HF model is::

    model(
        hidden_states: Tensor,       # [batch, seq, N_aux * hidden_size]
        inputs_embeds: Tensor,       # [batch, seq, hidden_size]
        attention_mask: Tensor | None,
    ) -> CausalLMOutput | Tensor

If the HF checkpoint returns a ``CausalLMOutput``, ``.logits`` is extracted.

FSDP notes:
  * Wrap only AFTER calling ``init_draft_model`` so the state dict has been
    loaded in full precision.
  * For models ≤ 3B params, ``ShardingStrategy.NO_SHARD`` (DDP-equivalent) is
    usually fastest since communication overhead is negligible.
"""

from __future__ import annotations

import logging
from typing import Dict, Generator, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EagleDraftModelWrapper(nn.Module):
    """Thin wrapper around a HF Eagle3 draft model for verl integration.

    Normalises the HF model's forward call so the rest of the training code
    only needs to call ``forward(hidden_states, inputs_embeds, attention_mask)``
    and receive logits as a plain tensor.
    """

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.model = hf_model

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the Eagle3 draft model forward pass.

        Args:
            hidden_states:  [batch, seq, N_aux * hidden_size]
                            Concatenated intermediate hidden states from the policy.
            inputs_embeds:  [batch, seq, hidden_size]
                            *Already rolled* (left-shifted by 1) policy input embeddings.
            attention_mask: Optional [batch, seq] boolean / float mask.

        Returns:
            logits: [batch, seq, vocab_size]
        """
        output = self.model(
            hidden_states=hidden_states,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        # HF models may return a dataclass (CausalLMOutput) or a plain tensor
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "logits"):
            return output.logits
        if isinstance(output, (tuple, list)):
            return output[0]

        raise TypeError(
            f"Unexpected output type from Eagle draft model: {type(output)}. "
            "Expected Tensor, CausalLMOutput, or tuple."
        )

    def named_parameters_for_optimizer(self) -> Generator[Tuple[str, nn.Parameter], None, None]:
        """Yield (name, param) pairs that require gradient updates."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield name, param


def load_eagle_draft_model(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Optional[str] = None,
) -> EagleDraftModelWrapper:
    """Load an Eagle3 draft model from a HuggingFace checkpoint.

    The checkpoint must support ``trust_remote_code=True`` since Eagle3 uses
    a custom model class not in the standard transformers library.

    Args:
        model_path:   Local path or HuggingFace repo id
                      (e.g. ``"AngelSlim/Qwen3-1.7B_eagle3"``).
        torch_dtype:  Dtype to load the model in (default bfloat16).
        device_map:   Optional HF device map (e.g. ``"auto"``).
                      Leave as None when using FSDP — let the caller handle
                      device placement.

    Returns:
        EagleDraftModelWrapper ready for FSDP / DDP wrapping.
    """
    from transformers import AutoModelForCausalLM

    logger.info(f"Loading Eagle3 draft model from: {model_path}")

    kwargs: Dict = dict(
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if device_map is not None:
        kwargs["device_map"] = device_map

    hf_model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    hf_model.train()

    wrapper = EagleDraftModelWrapper(hf_model)
    logger.info(
        f"Eagle3 draft model loaded ({sum(p.numel() for p in wrapper.parameters()) / 1e6:.0f}M params)"
    )
    return wrapper


def maybe_wrap_fsdp(
    model: nn.Module,
    device_id: int,
    use_fsdp: bool = True,
) -> nn.Module:
    """Optionally wrap the draft model with FSDP.

    For small Eagle3 models (≤ 3B) NO_SHARD is generally fastest because
    the all-reduce cost is minimal.  For larger drafters, consider SHARD_GRAD_OP.

    Args:
        model:     The EagleDraftModelWrapper instance.
        device_id: Local GPU index.
        use_fsdp:  If False, just moves model to device without FSDP.

    Returns:
        Possibly FSDP-wrapped module.
    """
    model = model.to(f"cuda:{device_id}")

    if not use_fsdp:
        return model

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy

        model = FSDP(
            model,
            device_id=device_id,
            sharding_strategy=ShardingStrategy.NO_SHARD,
            use_orig_params=True,
        )
        logger.info("Eagle3 draft model wrapped in FSDP (NO_SHARD / DDP-equivalent)")
    except Exception as e:
        logger.warning(f"FSDP wrapping failed: {e}. Falling back to plain module.")

    return model


def get_draft_state_dict_for_vllm(draft_model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract the draft model's state dict in a format suitable for vLLM weight loading.

    If the model is FSDP-wrapped, this must be called inside an
    ``FSDP.summon_full_params`` context on rank 0.

    Returns:
        Dict[str, Tensor] with keys matching the HF model's parameter names.
    """
    # Unwrap FSDP / DDP to reach the original HF module
    inner = draft_model
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(inner, FSDP):
            inner = inner.module
    except ImportError:
        pass

    from torch.nn.parallel import DistributedDataParallel

    if isinstance(inner, DistributedDataParallel):
        inner = inner.module

    # EagleDraftModelWrapper stores the HF model as `.model`
    if isinstance(inner, EagleDraftModelWrapper):
        inner = inner.model

    return {k: v.cpu() for k, v in inner.state_dict().items()}
