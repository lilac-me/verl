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

"""Eagle3 draft model: two construction paths.

Path A — load pretrained HF checkpoint (``model_path`` is set)
    ``load_eagle_draft_model()`` → ``EagleDraftModelWrapper``

Path B — build from policy components (``model_path`` is None)
    ``build_eagle3_from_policy()`` → ``Eagle3DraftModel``

Both expose the same forward signature::

    logits = draft_model(hidden_states, inputs_embeds, attention_mask)

eagle_output_layer design
--------------------------
``Eagle3DraftModel.eagle_output_layer`` is a Megatron
``ColumnParallelLinear(gather_output=True)`` with ``requires_grad=False``:

* Each TP rank stores only the local weight shard ``[vocab / TP, H]``.
* ``forward`` computes local logits ``[B, S, vocab / TP]`` then calls
  ``gather_from_tensor_model_parallel_region`` to produce full-vocabulary
  logits ``[B, S, vocab]``. Autograd is handled by Megatron's custom Function
  (backward is a reduce-scatter). TP = 1 is a no-op.
* ``sync_lm_head`` copies the policy's local lm-head shard directly — no
  all-gather needed because both sides hold the same-shaped local shard.
* ``get_draft_state_dict_for_vllm`` all-gathers the shard before yielding so
  that vLLM receives a complete weight tensor.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _gather_tp_weight(local_weight: torch.Tensor) -> torch.Tensor:
    """All-gather a TP-sharded weight tensor along dim 0.

    Used only when exporting weights to vLLM, which requires un-sharded tensors.
    TP = 1 returns the tensor unchanged.
    """
    import torch.distributed as dist
    from megatron.core import parallel_state

    if (
        parallel_state.model_parallel_is_initialized()
        and dist.is_available()
        and dist.is_initialized()
    ):
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            tp_group = parallel_state.get_tensor_model_parallel_group()
            shards = [torch.empty_like(local_weight) for _ in range(tp_size)]
            dist.all_gather(shards, local_weight.contiguous(), group=tp_group)
            return torch.cat(shards, dim=0).contiguous()

    return local_weight


# ---------------------------------------------------------------------------
# Path A: load pretrained HF Eagle3 checkpoint
# ---------------------------------------------------------------------------

class EagleDraftModelWrapper(nn.Module):
    """Thin wrapper around a pretrained HuggingFace Eagle3 model.

    The underlying model can be any EAGLE3-compatible architecture loadable
    via ``AutoModel.from_pretrained``.
    """

    def __init__(self, draft_model: nn.Module):
        super().__init__()
        self.draft_model = draft_model

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.draft_model(
            hidden_states=hidden_states,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        if isinstance(outputs, torch.Tensor):
            return outputs
        return outputs.logits


def load_eagle_draft_model(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> EagleDraftModelWrapper:
    """Load a pretrained Eagle3 draft model from a HuggingFace checkpoint."""
    from transformers import AutoModel, AutoModelForCausalLM

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    logger.info(f"Loading Eagle3 draft model from {model_path}")
    try:
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch_dtype, trust_remote_code=True
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, trust_remote_code=True
        )

    model = model.to(device).train()
    return EagleDraftModelWrapper(model)


# ---------------------------------------------------------------------------
# Path B: build Eagle3 draft model from policy components
# ---------------------------------------------------------------------------

class Eagle3DraftModel(nn.Module):
    """Eagle3 draft model assembled from policy components.

    Architecture::

        Input: [h_aux_1 ‖ … ‖ h_aux_N ‖ embed(t+1)]   (concatenated)
          → fc                : Linear((N_aux+1)*H → H, bias=False)
          → eagle_module      : HF transformer (num_draft_layers layers)
          → eagle_output_layer: ColumnParallelLinear(gather_output=True) — frozen, TP-sharded

    Attributes:
        fc                 : feature-fusion linear.
        eagle_module       : shallow HF transformer.
        eagle_output_layer : frozen Megatron ColumnParallelLinear (gather_output=True).
        _policy_lm_head_ref: reference to the policy's LM head used by
            ``sync_lm_head``. Set by ``build_eagle3_from_policy`` after construction.
    """

    def __init__(
        self,
        fc: nn.Linear,
        eagle_module: nn.Module,
        eagle_output_layer: nn.Module,
    ):
        super().__init__()
        self.fc = fc
        self.eagle_module = eagle_module
        self.eagle_output_layer = eagle_output_layer

        self._policy_lm_head_ref: Optional[nn.Module] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states:  [B, S, N_aux * H]  — concatenated policy aux states
            inputs_embeds:  [B, S, H]           — rolled token embeddings
            attention_mask: [B, S] bool, optional

        Returns:
            [B, S, vocab_size] logits (full vocabulary, gathered across TP)
        """
        x = self.fc(torch.cat([hidden_states, inputs_embeds], dim=-1))

        outputs = self.eagle_module(inputs_embeds=x, attention_mask=attention_mask)
        x = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        # ColumnParallelLinear(gather_output=True): returns (logits, bias)
        logits, _ = self.eagle_output_layer(x)
        return logits

    def sync_lm_head(self, policy_lm_head: Optional[nn.Module] = None) -> None:
        """Copy the policy's local lm-head shard into eagle_output_layer.

        Both the policy's ``output_layer`` (Megatron ColumnParallelLinear) and
        ``eagle_output_layer`` hold the same-shaped local shard ``[vocab / TP, H]``,
        so a direct copy suffices — no all-gather.

        Should be called after each policy optimizer step.
        """
        src = policy_lm_head or self._policy_lm_head_ref
        if src is None:
            return
        src_weight = getattr(src, "weight", None)
        if src_weight is None:
            return
        with torch.no_grad():
            self.eagle_output_layer.weight.copy_(
                src_weight.to(
                    device=self.eagle_output_layer.weight.device,
                    dtype=self.eagle_output_layer.weight.dtype,
                )
            )


def build_eagle3_from_policy(
    policy_model: nn.Module,
    hf_config,
    n_aux: int,
    num_draft_layers: int = 1,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> Eagle3DraftModel:
    """Build an Eagle3 draft model from the policy model's components.

    Does NOT require a pretrained Eagle3 checkpoint.

    Steps:
    1. ``fc``: newly initialised ``Linear((n_aux+1)*H → H, bias=False)``.
    2. ``eagle_module``: HF transformer with ``num_draft_layers`` layers,
       created from ``hf_config`` via ``AutoModel.from_config``, then
       initialised from the policy's last ``num_draft_layers`` decoder layers
       (``strict=False`` so mismatching keys are skipped gracefully).
    3. ``eagle_output_layer``: Megatron ``ColumnParallelLinear(gather_output=True)``
       initialised from the policy's local lm-head weight shard.  No all-gather —
       both sides hold ``[vocab / TP, H]`` shards on the same TP rank.
       Weight is frozen (``requires_grad=False``).

    Args:
        policy_model:     The policy ``nn.Module`` (unwrapped, single chunk).
        hf_config:        HuggingFace config object for the policy architecture.
        n_aux:            Number of auxiliary layers captured by hooks.
        num_draft_layers: Number of transformer layers in the draft model.
        torch_dtype:      Weight dtype (default bfloat16).
        device:           Target CUDA device.
    """
    from megatron.core.tensor_parallel import ColumnParallelLinear
    from transformers import AutoModel

    from verl.workers.eagle.hidden_capture import HiddenStateCapture

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    hidden_size = hf_config.hidden_size

    # ------------------------------------------------------------------
    # 1. Feature-fusion projection
    # ------------------------------------------------------------------
    fc = nn.Linear((n_aux + 1) * hidden_size, hidden_size, bias=False)
    nn.init.normal_(fc.weight, std=0.02)
    fc = fc.to(dtype=torch_dtype, device=device)

    # ------------------------------------------------------------------
    # 2. Shallow HF transformer base
    # ------------------------------------------------------------------
    draft_config = deepcopy(hf_config)
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.use_cache = False

    base = AutoModel.from_config(draft_config)
    base = base.to(dtype=torch_dtype, device=device).train()

    try:
        policy_layers = HiddenStateCapture._find_layers(policy_model)
        base_layers = HiddenStateCapture._find_layers(base)
        n = len(base_layers)
        for i, base_layer in enumerate(base_layers):
            src_layer = policy_layers[-(n - i)]
            missing, _ = base_layer.load_state_dict(src_layer.state_dict(), strict=False)
            if missing:
                logger.debug(f"Eagle3 draft layer {i}: missing keys {missing[:3]}…")
        logger.info(
            f"Eagle3 draft: initialised {n} layer(s) from policy's last {n} decoder layer(s)."
        )
    except Exception as exc:
        logger.warning(
            f"Eagle3 draft: layer init from policy failed ({exc}), using random weights."
        )

    # ------------------------------------------------------------------
    # 3. ColumnParallelLinear — frozen, TP-sharded, init from policy shard
    #
    # Both policy.output_layer (ColumnParallelLinear) and eagle_output_layer
    # hold [vocab/TP, H] on each TP rank, so we copy the local shard directly
    # without any all-gather.
    # ------------------------------------------------------------------
    policy_lm_head = HiddenStateCapture._find_lm_head(policy_model)
    if policy_lm_head is None:
        raise RuntimeError(
            "Cannot locate LM head in policy model for Eagle3 draft initialisation. "
            "Expected model.lm_head (HF) or model.output_layer (Mcore)."
        )

    lm_head = ColumnParallelLinear(
        input_size=hidden_size,
        output_size=hf_config.vocab_size,
        bias=False,
        gather_output=True,
        params_dtype=torch_dtype,
    )
    lm_head = lm_head.to(device=device)
    lm_head.weight.requires_grad_(False)
    with torch.no_grad():
        src_weight = getattr(policy_lm_head, "weight")
        lm_head.weight.copy_(src_weight.to(dtype=torch_dtype, device=device))

    model = Eagle3DraftModel(fc=fc, eagle_module=base, eagle_output_layer=lm_head)
    model._policy_lm_head_ref = policy_lm_head

    return model


# ---------------------------------------------------------------------------
# Weight export (shared by both paths)
# ---------------------------------------------------------------------------

def get_draft_state_dict_for_vllm(
    draft_model: nn.Module,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield ``(name, cpu_float32_tensor)`` pairs for loading into vLLM.

    For ``EagleDraftModelWrapper`` (Path A), the inner HF model's keys are
    yielded directly.

    For ``Eagle3DraftModel`` (Path B), ``eagle_output_layer.weight`` is
    all-gathered across TP ranks before yielding so that vLLM receives the
    complete ``[vocab, H]`` tensor.  All other keys are yielded as-is.
    """
    if isinstance(draft_model, EagleDraftModelWrapper):
        src = draft_model.draft_model
    else:
        src = draft_model

    for name, param in src.state_dict().items():
        tensor = param.detach()
        if isinstance(src, Eagle3DraftModel) and name == "eagle_output_layer.weight":
            tensor = _gather_tp_weight(tensor)
        yield name, tensor.cpu().float()
