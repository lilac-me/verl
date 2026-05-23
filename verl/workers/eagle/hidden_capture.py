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

"""Captures intermediate hidden states from the policy model for Eagle3 draft training.

Registers persistent forward hooks on selected decoder layers, the embedding
layer, and the LM head.  After the policy forward pass completes,
``get_captured_states()`` assembles the captured tensors into:

* ``hidden_states``:   concatenation of aux-layer outputs  [batch, seq, N_aux * hidden]
* ``inputs_embeds``:   embedding-layer output              [batch, seq, hidden]
* ``lm_head_logits``:  LM-head output (teacher for distil) [batch, seq, vocab]

Supported backends
------------------
* **HuggingFace / FSDP**: ``model.model.embed_tokens``, ``model.model.layers``,
  ``model.lm_head``
* **Megatron-Core (mcore), PP=1**: ``model.embedding.word_embeddings``,
  ``model.decoder.layers``, ``model.output_layer``

Pipeline parallelism (PP > 1) is NOT supported: each pipeline stage holds only
a subset of layers, so all required activations cannot be collected on a single
rank without explicit inter-stage communication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Eagle3 auxiliary-layer index heuristic (mirrors nemo-rl)
# ---------------------------------------------------------------------------

def get_eagle3_aux_layer_indices(num_layers: int) -> Tuple[int, ...]:
    """Return default aux-layer indices for Eagle3.

    Uses the nemo-rl heuristic: (1, num_layers // 2 - 1, num_layers - 4),
    deduplicated and sorted.
    """
    candidates = (
        1,
        max(0, num_layers // 2 - 1),
        max(1, num_layers - 4),
    )
    return tuple(sorted(set(candidates)))


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class CapturedStates:
    """Tensors captured from the policy model during a single forward pass."""

    # [batch, seq, N_aux * hidden_size]
    hidden_states: Optional[torch.Tensor] = None
    # [batch, seq, hidden_size]
    inputs_embeds: Optional[torch.Tensor] = None
    # [batch, seq, vocab_size]  — detached, float32
    lm_head_logits: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Embedding roll helper
# ---------------------------------------------------------------------------

def roll_inputs_embeds(embeds: torch.Tensor) -> torch.Tensor:
    """Left-shift embeddings by one token for Eagle3 time-step alignment.

    At position t the draft model predicts the policy distribution at t+1.
    Rolling the embeddings left aligns draft input at t with teacher output at
    t+1 (mirrors ``inputs_embeds[t+1]`` into position t).
    """
    return torch.roll(embeds, shifts=-1, dims=1)


# ---------------------------------------------------------------------------
# Main capture class
# ---------------------------------------------------------------------------

class HiddenStateCapture:
    """Registers persistent forward hooks on the policy model to collect activations.

    Supports both HuggingFace (FSDP) and Megatron-Core model layouts.  The
    hooks fire on every policy forward pass; captured data is cleared by the
    caller after each use to prevent stale state from leaking across steps.

    Args:
        model:             The policy nn.Module (or list[GPTModel] for Megatron).
        aux_layer_indices: Indices of decoder layers to capture.  None → auto.
        capture_logits:    Whether to hook the LM head for teacher logits.
    """

    def __init__(
        self,
        model: nn.Module,
        aux_layer_indices: Optional[Tuple[int, ...]] = None,
        capture_logits: bool = True,
    ):
        self._inner = self._unwrap(model)
        self._layers = self._find_layers(self._inner)
        self._embed = self._find_embed(self._inner)
        self._lm_head = self._find_lm_head(self._inner) if capture_logits else None

        num_layers = len(self._layers)
        self._aux_indices: Tuple[int, ...] = (
            aux_layer_indices
            if aux_layer_indices is not None
            else get_eagle3_aux_layer_indices(num_layers)
        )

        self._captured: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    # ------------------------------------------------------------------
    # Module discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap(model) -> nn.Module:
        """Unwrap FSDP / DDP / Megatron-list / torch.compile wrappers."""
        # Megatron: engine.module is a list[GPTModel] (one chunk per VPP stage)
        if isinstance(model, list):
            if not model:
                raise ValueError("Empty model list passed to HiddenStateCapture")
            model = model[0]

        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            while isinstance(model, FSDP):
                model = model.module
        except ImportError:
            pass

        from torch.nn.parallel import DistributedDataParallel as DDP
        while isinstance(model, DDP):
            model = model.module

        if hasattr(model, "_orig_mod"):  # torch.compile
            model = model._orig_mod

        return model

    @staticmethod
    def _find_layers(inner: nn.Module) -> nn.ModuleList:
        """Locate the decoder layer ModuleList.

        Tries HF paths first (``model.model.layers``, ``model.layers``), then
        Mcore path (``model.decoder.layers`` inside a TransformerBlock).
        """
        # HF-style
        for parent_attr in ("model", ""):
            obj = getattr(inner, parent_attr, inner) if parent_attr else inner
            for layers_attr in ("layers", "h", "blocks"):
                candidate = getattr(obj, layers_attr, None)
                if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
                    return candidate

        # Mcore-style: model.decoder is a TransformerBlock; layers live inside it
        decoder_block = getattr(inner, "decoder", None)
        if decoder_block is not None:
            candidate = getattr(decoder_block, "layers", None)
            if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
                return candidate

        raise AttributeError(
            "Cannot locate decoder layers. "
            "Expected model.model.layers (HF) or model.decoder.layers (Mcore)."
        )

    @staticmethod
    def _find_embed(inner: nn.Module) -> nn.Module:
        """Locate the token embedding module.

        Tries HF paths first, then Mcore ``model.embedding.word_embeddings``.
        """
        for path in (
            ("model", "embed_tokens"),           # LLaMA / Mistral / Qwen HF
            ("embed_tokens",),
            ("model", "wte"),                     # GPT-2 HF
            ("wte",),
            ("embedding", "word_embeddings"),     # Mcore GPTModel
            ("model", "embedding"),
            ("embedding",),
        ):
            obj = inner
            for part in path:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            else:
                if obj is not None:
                    return obj

        raise AttributeError(
            "Cannot locate embedding layer. "
            "Expected model.model.embed_tokens (HF) or "
            "model.embedding.word_embeddings (Mcore)."
        )

    @staticmethod
    def _find_lm_head(inner: nn.Module) -> Optional[nn.Module]:
        """Locate the LM head (linear projection to vocab size).

        Checks HF ``lm_head`` and Mcore ``output_layer``.
        """
        for attr in ("lm_head", "output_layer", "embed_out", "output", "head"):
            candidate = getattr(inner, attr, None)
            if candidate is not None and isinstance(candidate, nn.Module):
                return candidate
        return None

    # ------------------------------------------------------------------
    # Hook construction
    # ------------------------------------------------------------------

    def _make_layer_hook(self, layer_idx: int):
        def hook(_module, _args, output):
            # HF layers return (hidden, ...); Mcore layers return hidden directly
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or not hidden.is_floating_point():
                return
            # Accept [batch, seq, hidden] (bshd) and [total_tokens, hidden] (thd)
            if hidden.dim() in (2, 3):
                self._captured[f"layer_{layer_idx}"] = hidden.detach()
        return hook

    def _make_embed_hook(self):
        def hook(_module, _args, output):
            embeds = output[0] if isinstance(output, tuple) else output
            if not isinstance(embeds, torch.Tensor) or not embeds.is_floating_point():
                return
            self._captured["embeds"] = embeds.detach()
        return hook

    def _make_lm_head_hook(self):
        def hook(_module, _args, output):
            logits = output[0] if isinstance(output, tuple) else output
            if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
                return
            # Keep in float32 for numerically stable soft cross-entropy
            self._captured["lm_head_logits"] = logits.detach().float()
        return hook

    # ------------------------------------------------------------------
    # Hook lifecycle
    # ------------------------------------------------------------------

    def register_hooks(self) -> None:
        """Register persistent hooks on the policy model.

        Called once during initialization; hooks remain active for the
        lifetime of the manager.
        """
        self._hooks.clear()
        self._hooks.append(self._embed.register_forward_hook(self._make_embed_hook()))

        for idx in self._aux_indices:
            if idx < len(self._layers):
                self._hooks.append(
                    self._layers[idx].register_forward_hook(self._make_layer_hook(idx))
                )

        if self._lm_head is not None:
            self._hooks.append(self._lm_head.register_forward_hook(self._make_lm_head_hook()))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # State assembly
    # ------------------------------------------------------------------

    def get_captured_states(self) -> CapturedStates:
        """Assemble captured tensors into a CapturedStates container.

        Must be called after the policy forward pass and before _captured is
        cleared.
        """
        embeds = self._captured.get("embeds")

        chunks: List[torch.Tensor] = []
        for idx in sorted(self._aux_indices):
            t = self._captured.get(f"layer_{idx}")
            if t is not None:
                chunks.append(t)

        lm_head_logits = self._captured.get("lm_head_logits")

        if not chunks:
            return CapturedStates(
                hidden_states=None,
                inputs_embeds=embeds,
                lm_head_logits=lm_head_logits,
            )

        return CapturedStates(
            hidden_states=torch.cat(chunks, dim=-1),  # [batch, seq, N*hidden]
            inputs_embeds=embeds,
            lm_head_logits=lm_head_logits,
        )
