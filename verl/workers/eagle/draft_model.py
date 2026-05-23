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

lm_head gradient semantics (Path B)
------------------------------------
The draft's lm_head is a *frozen deep copy* of the policy's lm_head
(``requires_grad=False`` on all its parameters).

When ``requires_grad=False``, PyTorch does **not** accumulate ``.grad`` on
those parameters, but it **does** propagate ``∂L/∂(lm_head_input)`` backwards
through the weight matrix via the chain rule.  This means ``fc`` and the base
transformer layers receive correct gradients and can be updated, while the
lm_head parameters stay unchanged.

Because the frozen lm_head drifts from the policy's lm_head as RL training
progresses, ``Eagle3DraftModel.sync_lm_head()`` should be called after each
policy optimizer step to copy the latest policy lm_head weights into the
draft's frozen copy.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path A: load pretrained HF Eagle3 checkpoint
# ---------------------------------------------------------------------------

class EagleDraftModelWrapper(nn.Module):
    """Thin wrapper around a pretrained HuggingFace Eagle3 model.

    The underlying model can be any EAGLE3-compatible architecture loadable
    via ``AutoModel.from_pretrained``.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.model(
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
          → fc : Linear((N_aux+1)*H → H, bias=False)
          → base: HF transformer (num_draft_layers, init from policy last layers)
          → lm_head: frozen deep copy of policy's LM head

    Attributes:
        fc       : the feature-fusion linear.
        base     : the shallow HF transformer.
        lm_head  : frozen projection (``requires_grad=False``).
        _policy_lm_head_ref: weak reference to the policy's LM head module used
            to sync ``lm_head`` after each policy optimizer step. Set by
            ``EagleDraftManager`` after construction.
    """

    def __init__(self, fc: nn.Linear, base: nn.Module, lm_head: nn.Module):
        super().__init__()
        self.fc = fc
        self.base = base
        self.lm_head = lm_head  # frozen; in state_dict, excluded from optimizer

        # Set by EagleDraftManager so we can sync after policy steps
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
            [B, S, vocab_size] logits
        """
        # Feature fusion: (N_aux+1)*H → H
        x = self.fc(torch.cat([hidden_states, inputs_embeds], dim=-1))

        # Shallow transformer (inputs_embeds path bypasses token embedding)
        outputs = self.base(inputs_embeds=x, attention_mask=attention_mask)
        x = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        # Vocab projection (frozen; gradients flow through via chain rule)
        return self.lm_head(x)

    def sync_lm_head(self, policy_lm_head: Optional[nn.Module] = None) -> None:
        """Copy policy's updated LM-head weights into this frozen copy.

        Should be called after each policy optimizer step so the draft's output
        projection stays aligned with the evolving policy vocabulary.

        Args:
            policy_lm_head: The policy's LM head module.  If None, falls back
                to ``self._policy_lm_head_ref``.
        """
        src = policy_lm_head or self._policy_lm_head_ref
        if src is None:
            return
        with torch.no_grad():
            for dst_p, src_p in zip(self.lm_head.parameters(), src.parameters()):
                dst_p.copy_(src_p)


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
    2. ``base``: HF transformer with ``num_draft_layers`` layers, created from
       ``hf_config`` via ``AutoModel.from_config``, then initialised from the
       policy's **last** ``num_draft_layers`` decoder layers (``strict=False``
       so mismatching keys are skipped gracefully).
    3. ``lm_head``: frozen deep copy of the policy's LM head.

    Args:
        policy_model:     The policy ``nn.Module`` (unwrapped, single chunk).
        hf_config:        HuggingFace config object for the policy architecture.
        n_aux:            Number of auxiliary layers captured by hooks.
        num_draft_layers: Number of transformer layers in the draft model.
        torch_dtype:      Weight dtype (default bfloat16).
        device:           Target CUDA device.
    """
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
    draft_config.use_cache = False  # no KV cache needed during training

    base = AutoModel.from_config(draft_config)
    base = base.to(dtype=torch_dtype, device=device).train()

    # Initialise base layers from the policy's last N decoder layers.
    # strict=False lets mismatching keys (e.g. rotary buffer shapes) be skipped.
    try:
        policy_layers = HiddenStateCapture._find_layers(policy_model)
        base_layers = HiddenStateCapture._find_layers(base)
        n = len(base_layers)
        for i, base_layer in enumerate(base_layers):
            src_layer = policy_layers[-(n - i)]
            missing, unexpected = base_layer.load_state_dict(
                src_layer.state_dict(), strict=False
            )
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
    # 3. Frozen LM head (deep copy, requires_grad=False)
    # ------------------------------------------------------------------
    policy_lm_head = HiddenStateCapture._find_lm_head(policy_model)
    if policy_lm_head is None:
        raise RuntimeError(
            "Cannot locate LM head in policy model for Eagle3 draft initialisation. "
            "Expected model.lm_head (HF) or model.output_layer (Mcore)."
        )

    lm_head = deepcopy(policy_lm_head).to(dtype=torch_dtype, device=device)
    for p in lm_head.parameters():
        p.requires_grad = False

    model = Eagle3DraftModel(fc=fc, base=base, lm_head=lm_head)

    # Store a reference so sync_lm_head() can be called without arguments
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

    For ``Eagle3DraftModel`` (Path B), the full state dict is yielded — vLLM
    must be configured with the matching Eagle3 architecture to accept these
    keys.
    """
    if isinstance(draft_model, EagleDraftModelWrapper):
        src = draft_model.model
    else:
        src = draft_model

    for name, param in src.state_dict().items():
        yield name, param.detach().cpu().float()
